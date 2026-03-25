import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention import get_anneal_attn_mask
from data import build_packed_dataloader
from eval import run_evals


def diffusion_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mask_token_id: int,
    shift: bool,
    global_step: int,
    anneal_steps: int,
    attn_impl: str,
) -> torch.Tensor:
    """
    One diffusion training step on a batch of full sequences.

    - We sample continuous time t ~ Uniform(epsilon, 1), set sigma=t, and mask tokens
      with probability sigma (absorbing diffusion).
    - src_mask marks tokens that should never be masked (all False here).
    - Loss is cross-entropy on masked positions only.
    - dsigma = 1/t weights earlier timesteps more; you can drop this weighting
      if you want an unweighted objective.
    - If shift=True, we align predictions to next-token positions (AR-style).
    - Annealed attention gradually relaxes causality from causal -> bidirectional
      over `anneal_steps`.
    """
    batch_size, seq_len = input_ids.shape
    src_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)

    sampling_eps = 1e-3
    t = (1 - sampling_eps) * torch.rand(batch_size, device=input_ids.device) + sampling_eps
    sigma = t
    dsigma = torch.reciprocal(t)

    to_demask = ~src_mask
    move_indices = (torch.rand(*input_ids.shape, device=input_ids.device) < sigma[:, None]) & to_demask
    x_t = torch.where(move_indices, mask_token_id, input_ids)

    attention_mask = None
    if attn_impl == "sdpa":
        attn_mask_ratio = 1.0
        if anneal_steps > 0:
            attn_mask_ratio = min(1.0, float(global_step + 1) / float(anneal_steps))
        param = next(model.parameters())
        attention_mask = get_anneal_attn_mask(
            seq_len,
            batch_size,
            dtype=param.dtype,
            device=param.device,
            attn_mask_ratio=attn_mask_ratio,
        )
    logits = model(input_ids=x_t, attention_mask=attention_mask, use_cache=False, return_dict=True).logits

    loss_mask = x_t == mask_token_id
    targets = input_ids

    if shift:
        logits = logits[:, :-1]
        loss_mask = loss_mask[:, 1:]
        targets = input_ids[:, 1:]

    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(batch_size, -1)
    loss = loss.masked_fill(~loss_mask, 0)
    loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()
    return loss


def parse_comma_separated(value: Optional[str]) -> Optional[list[str]]:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def resolve_checkpoint_dir(args: argparse.Namespace) -> Optional[Path]:
    if args.checkpoint_dir is not None:
        return Path(args.checkpoint_dir)
    if args.output_dir is not None:
        return Path(args.output_dir) / "checkpoints"
    return None


def resolve_eval_history_path(args: argparse.Namespace) -> Optional[Path]:
    if args.eval_history_path is not None:
        return Path(args.eval_history_path)
    if args.output_dir is not None:
        return Path(args.output_dir) / "eval_history.jsonl"
    return None


def append_jsonl_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer,
    checkpoint_dir: Optional[Path],
    global_step: int,
) -> Optional[Path]:
    if checkpoint_dir is None:
        return None

    checkpoint_path = checkpoint_dir / f"step_{global_step:08d}"
    if accelerator.is_main_process:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    accelerator.wait_for_everyone()
    return checkpoint_path


def maybe_init_wandb(accelerator: Accelerator, args: argparse.Namespace) -> None:
    if not args.wandb_project:
        return

    config = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value

    init_kwargs = {
        "wandb": {
            "name": args.wandb_run_name,
            "entity": args.wandb_entity,
            "dir": args.wandb_dir,
        }
    }
    accelerator.init_trackers(project_name=args.wandb_project, config=config, init_kwargs=init_kwargs)


def maybe_log_metrics(accelerator: Accelerator, args: argparse.Namespace, metrics: dict[str, float], step: int) -> None:
    if args.wandb_project and metrics:
        accelerator.log(metrics, step=step)


def run_scheduled_eval(
    accelerator: Accelerator,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tokenizer,
    global_step: int,
    checkpoint_path: Optional[Path],
    eval_history_path: Optional[Path],
) -> None:
    if args.eval_every <= 0 or global_step % args.eval_every != 0:
        return

    accelerator.wait_for_everyone()
    metrics = None

    if accelerator.is_main_process:
        eval_model = accelerator.unwrap_model(model)
        eval_model.eval()
        metrics = run_evals(
            model=eval_model,
            tokenizer=tokenizer,
            tasks=parse_comma_separated(args.eval_tasks) or [],
            diffusion_steps=args.eval_diffusion_steps,
            shift=args.shift,
            max_samples=args.eval_max_samples,
            lambada_path=Path(args.lambada_path),
            poem_path=Path(args.poem_path),
            poem_direction=args.poem_direction,
            gen_length=args.gen_length,
            print_results=False,
        )
        metric_str = ", ".join(f"{name}={value:.4f}" for name, value in sorted(metrics.items()))
        accelerator.print(f"step {global_step} | eval | {metric_str}")

        if eval_history_path is not None:
            append_jsonl_record(
                eval_history_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "step": global_step,
                    "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
                    "metrics": metrics,
                },
            )

    accelerator.wait_for_everyone()

    if metrics is not None:
        maybe_log_metrics(
            accelerator,
            args,
            {f"eval/{name}": value for name, value in metrics.items()},
            step=global_step,
        )

    model.train()


def main(args: argparse.Namespace) -> None:
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb_project else None,
        kwargs_handlers=[timeout],
    )
    accelerator.print(f"Total processes: {accelerator.num_processes}")
    maybe_init_wandb(accelerator, args)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation=args.attn_impl,
    )
    if args.attn_impl == "flash_attention_2":
        for layer in model.model.layers:
            layer.self_attn.is_causal = False

    added_tokens = 0
    if tokenizer.mask_token is None:
        added_tokens += tokenizer.add_special_tokens({"mask_token": args.mask_token})
    if tokenizer.pad_token is None:
        pad = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.mask_token
        added_tokens += tokenizer.add_special_tokens({"pad_token": pad})
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)

    effective_block_size = args.seq_length + (1 if args.shift else 0)
    prefixes = parse_comma_separated(args.packed_prefixes)
    weights = [float(w) for w in parse_comma_separated(args.packed_weights)] if args.packed_weights else None
    train_loader = build_packed_dataloader(
        Path(args.packed_data_dir),
        prefixes=prefixes,
        weights=weights,
        batch_size=args.batch_size,
        block_size=effective_block_size,
        num_processes=accelerator.num_processes,
        process_rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    optimizer.zero_grad()

    checkpoint_dir = resolve_checkpoint_dir(args)
    eval_history_path = resolve_eval_history_path(args)

    model.train()
    global_step = 0
    for _, batch in enumerate(train_loader):
        input_ids = batch
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        with accelerator.accumulate(model):
            loss = diffusion_step(
                model,
                input_ids,
                tokenizer.mask_token_id,
                shift=args.shift,
                global_step=global_step,
                anneal_steps=args.anneal_steps,
                attn_impl=args.attn_impl,
            )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                reduced_loss = accelerator.gather_for_metrics(loss.detach().float().reshape(1)).mean().item()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                accelerator.print(f"step {global_step} | loss {reduced_loss:.4f}")
                maybe_log_metrics(accelerator, args, {"train/loss": reduced_loss}, step=global_step)

                checkpoint_path = None
                if args.save_every > 0 and global_step % args.save_every == 0:
                    checkpoint_path = save_checkpoint(accelerator, model, tokenizer, checkpoint_dir, global_step)

                run_scheduled_eval(
                    accelerator,
                    args,
                    model,
                    tokenizer,
                    global_step,
                    checkpoint_path,
                    eval_history_path,
                )

            if global_step >= args.max_train_steps:
                break

    accelerator.print("Training finished")

    if args.output_dir is not None and accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulate-every", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--anneal-steps", type=int, default=1000)
    parser.add_argument("--mask-token", type=str, default="[MASK]")
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")

    parser.add_argument("--packed-data-dir", type=str, required=True)
    parser.add_argument("--packed-prefixes", type=str, default=None)
    parser.add_argument("--packed-weights", type=str, default=None)

    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-history-path", type=str, default=None)
    parser.add_argument("--eval-tasks", type=str, default="hellaswag,piqa,lambada,winogrande,piqa,siqa,poem_reverse")
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--eval-diffusion-steps", type=int, default=64)
    parser.add_argument("--lambada-path", type=str, default="eval_data/lambada_test_plain_text.txt")
    parser.add_argument("--poem-path", type=str, default="eval_data/poem_data.json")
    parser.add_argument("--poem-direction", type=str, default="ftb", choices=["ftb", "btf"])
    parser.add_argument("--gen-length", type=int, default=28)
    parser.add_argument("--wandb-project", type=str, default="miniDiffuLLaMa")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-dir", type=str, default=None)

    main(parser.parse_args())
