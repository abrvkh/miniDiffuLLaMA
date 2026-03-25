import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention import get_anneal_attn_mask


def eval_forward(
    model,
    input_ids: torch.Tensor,
    src_mask: torch.Tensor,
    mask_token_id: int,
    diffusion_steps: int,
    shift: bool,
) -> torch.Tensor:
    """
    Scores a full sequence by denoising loss on the masked portion.

    We treat `src_mask=True` tokens as fixed context and only mask/predict tokens
    where src_mask is False (typically the candidate continuation). For each
    timestep t (from T to 1), we randomly mask tokens with rate t/T, run the
    model, and compute cross-entropy only on masked positions. The final score
    is the average masked-token loss across timesteps (lower is better).
    """
    model.eval()
    x = input_ids
    batch_size, seq_len = x.shape
    total_unw_loss = torch.tensor(0.0, device=x.device)

    for t in range(diffusion_steps, 0, -1):
        rate = t / diffusion_steps
        tt = torch.tensor([rate] * batch_size, device=x.device)
        sigma = tt
        dsigma = torch.reciprocal(tt)

        to_demask = ~src_mask
        move_indices = (torch.rand(*x.shape, device=x.device) < sigma[:, None]) & to_demask
        x_t = torch.where(move_indices, mask_token_id, x)

        param = next(model.parameters())
        attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=param.dtype, device=param.device, attn_mask_ratio=1.0)
        logits = model(input_ids=x_t, attention_mask=attention_mask, use_cache=False, return_dict=True).logits

        loss_mask = x_t == mask_token_id
        labels = x

        if shift:
            logits = logits[:, :-1]
            loss_mask = loss_mask[:, 1:]
            labels = x[:, 1:]

        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="none").reshape(batch_size, -1)
        loss = loss.masked_fill(~loss_mask, 0)
        if loss_mask.sum() == 0:
            continue
        unw_loss = loss.sum() / loss_mask.sum()
        total_unw_loss += unw_loss

    return total_unw_loss / diffusion_steps


@torch.no_grad()
def generate_solution(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    src_mask: torch.Tensor,
    diffusion_steps: int,
    shift: bool,
) -> torch.Tensor:
    """
    Iterative random unmasking sampler.

    Starts with all non-src tokens replaced by [MASK]. At each step, we sample
    a full x0 from the model and then randomly fix a fraction of remaining
    masked positions (p = 1/(t+1)).
    """
    model.eval()
    x = input_ids.to(model.device)
    src_mask = src_mask.to(model.device)

    seq_len = x.size(1)
    batch_size = x.size(0)
    param = next(model.parameters())
    attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=param.dtype, device=param.device, attn_mask_ratio=1.0)

    to_demask = ~src_mask
    xt = x.masked_fill(to_demask, tokenizer.mask_token_id)

    for t in range(diffusion_steps - 1, -1, -1):
        logits = model(input_ids=xt, attention_mask=attention_mask, use_cache=False, return_dict=True).logits
        scores = torch.log_softmax(logits, dim=-1)
        x0 = torch.distributions.Categorical(logits=scores).sample()

        if shift:
            x0 = torch.cat([x[:, 0:1], x0[:, :-1]], dim=1)

        p_to_fix = 1 / (t + 1)
        to_fix = to_demask & (torch.rand_like(x0, dtype=torch.float) < p_to_fix)
        xt = torch.where(to_fix, x0, xt)
        to_demask = to_demask.masked_fill(to_fix, False)

    if shift:
        xt = xt[:, 1:]

    return xt


def build_prompt_completion(
    tokenizer,
    prompt: str,
    gen_length: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Builds a prompt+mask input for generation.

    Returns:
    - input_ids: [1, prompt_len + gen_length] with [MASK] for the completion.
    - src_mask: True for prompt tokens, False for the completion.
    - prompt_len: length of the prompt in tokens.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = prompt_ids + [tokenizer.mask_token_id] * gen_length
    src_mask = [1] * len(prompt_ids) + [0] * gen_length
    input_ids_t = torch.tensor([input_ids], device=device, dtype=torch.long)
    src_mask_t = torch.tensor([src_mask], device=device, dtype=torch.bool)
    return input_ids_t, src_mask_t, len(prompt_ids)


def eval_hellaswag(model, tokenizer, diffusion_steps: int, shift: bool, max_samples: Optional[int]) -> float:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_samples is not None:
        ds = ds.select(range(max_samples))

    correct = 0
    total = 0
    for ex in ds:
        ctx = ex["ctx"]
        endings = ex["endings"]
        label = int(ex["label"])
        scores = []
        for ending in endings:
            text = ctx + " " + ending
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            prefix = tokenizer.encode(ctx, add_special_tokens=False)
            src_mask = [1] * len(prefix) + [0] * (len(input_ids) - len(prefix))
            inputs = torch.tensor([input_ids], device=model.device)
            src_mask = torch.tensor([src_mask], device=model.device, dtype=torch.bool)
            score = eval_forward(model, inputs, src_mask, tokenizer.mask_token_id, diffusion_steps, shift)
            scores.append(score.item())
        pred = int(min(range(len(scores)), key=lambda i: scores[i]))
        correct += int(pred == label)
        total += 1
    return correct / max(1, total)


def eval_lambada(model, tokenizer, diffusion_steps: int, shift: bool, lambada_path: Path) -> float:
    if not lambada_path.exists():
        raise FileNotFoundError(f"Lambada file not found: {lambada_path}")

    total_cnt = 0
    correct = 0
    with lambada_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_cnt += 1
            x0 = tokenizer.encode(line, add_special_tokens=False)
            prefix = tokenizer.encode(" ".join(line.split()[:-1]), add_special_tokens=False)
            masked_nums = len(x0) - len(prefix)
            src_mask = [1] * len(prefix) + [0] * masked_nums
            inputs = torch.tensor([x0], device=model.device)
            src_mask = torch.tensor([src_mask], device=model.device, dtype=torch.bool)
            res = generate_solution(model, tokenizer, inputs, src_mask, diffusion_steps=masked_nums, shift=shift)
            pred = tokenizer.decode(res.tolist()[0][-masked_nums:]).strip()
            if pred == line.split()[-1].strip():
                correct += 1
    return correct / max(1, total_cnt)


def eval_winogrande(model, tokenizer, diffusion_steps: int, shift: bool, max_samples: Optional[int]) -> float:
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(max_samples))

    correct = 0
    total = 0
    for ex in ds:
        prefix = ex["sentence"].replace("_", "")
        options = [ex["option1"], ex["option2"]]
        label = int(ex["answer"]) - 1
        scores = []
        for opt in options:
            text = prefix + " " + opt
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            src_mask = [1] * len(prefix_ids) + [0] * (len(input_ids) - len(prefix_ids))
            inputs = torch.tensor([input_ids], device=model.device)
            src_mask = torch.tensor([src_mask], device=model.device, dtype=torch.bool)
            score = eval_forward(model, inputs, src_mask, tokenizer.mask_token_id, diffusion_steps, shift)
            scores.append(score.item())
        pred = int(min(range(len(scores)), key=lambda i: scores[i]))
        correct += int(pred == label)
        total += 1
    return correct / max(1, total)


def eval_piqa(model, tokenizer, diffusion_steps: int, shift: bool, max_samples: Optional[int]) -> float:
    ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(max_samples))

    correct = 0
    total = 0
    for ex in ds:
        prefix = ex["goal"]
        options = [ex["sol1"], ex["sol2"]]
        label = int(ex["label"])
        scores = []
        for opt in options:
            text = prefix + " " + opt
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            src_mask = [1] * len(prefix_ids) + [0] * (len(input_ids) - len(prefix_ids))
            inputs = torch.tensor([input_ids], device=model.device)
            src_mask = torch.tensor([src_mask], device=model.device, dtype=torch.bool)
            score = eval_forward(model, inputs, src_mask, tokenizer.mask_token_id, diffusion_steps, shift)
            scores.append(score.item())
        pred = int(min(range(len(scores)), key=lambda i: scores[i]))
        correct += int(pred == label)
        total += 1
    return correct / max(1, total)


def eval_siqa(model, tokenizer, diffusion_steps: int, shift: bool, max_samples: Optional[int]) -> float:
    ds = load_dataset("allenai/social_i_qa", split="validation", trust_remote_code=True)
    if max_samples is not None:
        ds = ds.select(range(max_samples))

    correct = 0
    total = 0
    for ex in ds:
        prefix = ex["context"] + " " + ex["question"]
        options = [ex["answerA"], ex["answerB"], ex["answerC"]]
        label = ["A", "B", "C"].index(ex["label"])
        scores = []
        for opt in options:
            text = prefix + " " + opt
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            src_mask = [1] * len(prefix_ids) + [0] * (len(input_ids) - len(prefix_ids))
            inputs = torch.tensor([input_ids], device=model.device)
            src_mask = torch.tensor([src_mask], device=model.device, dtype=torch.bool)
            score = eval_forward(model, inputs, src_mask, tokenizer.mask_token_id, diffusion_steps, shift)
            scores.append(score.item())
        pred = int(min(range(len(scores)), key=lambda i: scores[i]))
        correct += int(pred == label)
        total += 1
    return correct / max(1, total)


def eval_poem_reverse(
    model,
    tokenizer,
    diffusion_steps: int,
    shift: bool,
    gen_length: int,
    poem_path: Path,
    direction: str,
) -> float:
    if not poem_path.exists():
        raise FileNotFoundError(f"Poem data not found: {poem_path}")
    with poem_path.open("r", encoding="utf-8") as f:
        poems = json.load(f)

    extra_prompt = " 直接输出句子即可。"

    if direction == "ftb":
        prompts = [poem["first"] + "的下一句是什么？" + extra_prompt for poem in poems]
        answers = [poem["second"] for poem in poems]
    elif direction == "btf":
        prompts = [poem["second"] + "的上一句是什么？" + extra_prompt for poem in poems]
        answers = [poem["first"] for poem in poems]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    acc = 0
    for prompt, answer in zip(prompts, answers):
        input_ids, src_mask, prompt_len = build_prompt_completion(
            tokenizer, prompt, gen_length=gen_length, device=model.device
        )
        out = generate_solution(model, tokenizer, input_ids, src_mask, diffusion_steps=diffusion_steps, shift=shift)
        gen_text = tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        if answer in gen_text:
            acc += 1
    return acc / max(1, len(prompts))


def run_evals(
    model,
    tokenizer,
    tasks: list[str],
    diffusion_steps: int,
    shift: bool,
    max_samples: Optional[int],
    lambada_path: Path,
    poem_path: Path,
    poem_direction: str,
    gen_length: int,
    print_results: bool = True,
) -> dict[str, float]:
    metrics = {}
    for task in tasks:
        if task == "lambada":
            metrics["lambada_acc"] = eval_lambada(
                model,
                tokenizer,
                diffusion_steps,
                shift,
                lambada_path=lambada_path,
            )
        elif task == "hellaswag":
            metrics["hellaswag_acc"] = eval_hellaswag(model, tokenizer, diffusion_steps, shift, max_samples)
        elif task == "winogrande":
            metrics["winogrande_acc"] = eval_winogrande(model, tokenizer, diffusion_steps, shift, max_samples)
        elif task == "piqa":
            metrics["piqa_acc"] = eval_piqa(model, tokenizer, diffusion_steps, shift, max_samples)
        elif task == "siqa":
            metrics["siqa_acc"] = eval_siqa(model, tokenizer, diffusion_steps, shift, max_samples)
        elif task == "poem_reverse":
            metrics["poem_reverse_acc"] = eval_poem_reverse(
                model,
                tokenizer,
                diffusion_steps,
                shift,
                gen_length=gen_length,
                poem_path=poem_path,
                direction=poem_direction,
            )
        else:
            raise ValueError(f"Unknown eval task: {task}")

    if print_results:
        for name, value in sorted(metrics.items()):
            print(f"{name}: {value:.6f}")
    return metrics


def load_eval_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Eval history not found: {path}")

        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            continue

        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            records.extend(payload)
        elif isinstance(payload, dict) and "records" in payload and isinstance(payload["records"], list):
            records.extend(payload["records"])
        else:
            raise ValueError(f"Unsupported eval history format in {path}")

    return sorted(records, key=lambda record: int(record.get("step", 0)))


def build_checkpoint_metric_table(records: list[dict]) -> tuple[list[str], list[list[str]]]:
    if not records:
        raise ValueError("No evaluation records found.")

    metric_names = sorted({name for record in records for name in record.get("metrics", {}).keys()})
    checkpoints = []
    for record in records:
        step = int(record.get("step", 0))
        checkpoint = record.get("checkpoint")
        label = Path(checkpoint).name if checkpoint else f"step_{step:08d}"
        checkpoints.append((label, record))

    headers = ["eval"] + [label for label, _ in checkpoints]
    rows = []
    for metric_name in metric_names:
        row = [metric_name]
        for _, record in checkpoints:
            value = record.get("metrics", {}).get(metric_name)
            row.append("" if value is None else f"{float(value):.4f}")
        rows.append(row)
    return headers, rows


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(str(cell)) for cell in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def render(row: list[str]) -> str:
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    print(render(headers))
    print(separator)
    for row in rows:
        print(render(row))


def print_eval_history_table(paths: list[Path]) -> None:
    headers, rows = build_checkpoint_metric_table(load_eval_records(paths))
    print_table(headers, rows)


def main(args: argparse.Namespace) -> None:
    if args.history_paths:
        paths = [Path(path.strip()) for path in args.history_paths.split(",") if path.strip()]
        print_eval_history_table(paths)
        return

    if args.model is None:
        raise ValueError("--model is required unless --history-paths is used.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        _attn_implementation=args.attn_impl,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    tasks = [t.strip() for t in args.eval_tasks.split(",") if t.strip()]
    metrics = run_evals(
        model,
        tokenizer,
        tasks,
        args.diffusion_steps,
        args.shift,
        args.eval_max_samples,
        lambada_path=Path(args.lambada_path),
        poem_path=Path(args.poem_path),
        poem_direction=args.poem_direction,
        gen_length=args.gen_length,
        print_results=True,
    )
    if args.results_json is not None:
        output_path = Path(args.results_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"step": args.step, "checkpoint": args.checkpoint_label, "metrics": metrics}
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--attn-impl", type=str, default="flash_attention_2")
    parser.add_argument("--diffusion-steps", type=int, default=64)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--eval-tasks", type=str, default="hellaswag,winogrande,piqa,siqa")
    parser.add_argument("--eval-max-samples", type=int, default=None)
    parser.add_argument("--lambada-path", type=str, default="eval_data/lambada_test_plain_text.txt")
    parser.add_argument("--poem-path", type=str, default="eval_data/poem_data.json")
    parser.add_argument("--poem-direction", type=str, default="ftb", choices=["ftb", "btf"])
    parser.add_argument("--gen-length", type=int, default=28)
    parser.add_argument("--results-json", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--checkpoint-label", type=str, default=None)
    parser.add_argument("--history-paths", type=str, default=None)
    main(parser.parse_args())
