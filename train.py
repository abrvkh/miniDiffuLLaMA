import argparse
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import build_packed_dataloader


def get_anneal_attn_mask(seq_len: int, bsz: int, dtype, device, attn_mask_ratio: float) -> torch.Tensor:
    """
    Builds a causal mask and gradually relaxes it with random extra attention
    according to attn_mask_ratio. Returns a 4D additive mask with 0 for allowed
    positions and -inf for blocked positions.
    """
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=dtype))
    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)
    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def diffusion_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    mask_token_id: int,
    shift: bool,
    global_step: int,
    anneal_steps: int,
) -> torch.Tensor:
    """
    One diffusion training step on a batch of full sequences.

    - We sample continuous time t ~ Uniform(ε, 1), set sigma=t, and mask tokens
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

    attn_mask_ratio = 1.0
    if anneal_steps > 0:
        attn_mask_ratio = min(1.0, float(global_step + 1) / float(anneal_steps))

    param = next(model.parameters())
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=param.dtype, device=param.device, attn_mask_ratio=attn_mask_ratio)
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


def main(args: argparse.Namespace) -> None:
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        kwargs_handlers=[timeout],
    )
    accelerator.print(f"Total processes: {accelerator.num_processes}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        _attn_implementation=args.attn_impl,
    )

    added_tokens = 0
    if tokenizer.mask_token is None:
        added_tokens += tokenizer.add_special_tokens({"mask_token": args.mask_token})
    if tokenizer.pad_token is None:
        pad = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.mask_token
        added_tokens += tokenizer.add_special_tokens({"pad_token": pad})
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)

    effective_block_size = args.seq_length + (1 if args.shift else 0)
    prefixes = [p.strip() for p in args.packed_prefixes.split(",")] if args.packed_prefixes else None
    weights = [float(w) for w in args.packed_weights.split(",")] if args.packed_weights else None
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

    model.train()
    global_step = 0
    for step, batch in enumerate(train_loader):
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
            )
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
                accelerator.print(f"step {global_step} | loss {loss.item():.4f}")
                global_step += 1

            if global_step >= args.max_train_steps:
                break

    accelerator.print("Training finished")

    if args.output_dir is not None and accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


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

    main(parser.parse_args())
