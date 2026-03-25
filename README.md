# miniDiffuLLaMA

Minimal diffusion-LM training and eval using TinyLLaMA-style packed datasets. E.g. it can use meta-llama/Llama-2-7b-hf autoregressive LLM to be changed into a diffusion-LM.

## Setup (uv)

This repo uses standard Python deps. With `uv`, you can create a venv and sync:

```bash
uv venv
source .venv/bin/activate

uv sync
```

If you need Flash Attention 2, depending on your system use:

```bash
export TORCH=torch2.6
export CXX=cxx11abiFALSE

uv pip install "flash-attn==2.7.4.post1" \
  --extra-index-url https://thomasjpfan.github.io/flash-attn-whl/cu12/$TORCH/$CXX \
  --no-cache-dir
```

## Data (packed .bin shards)

This pipeline expects pre-tokenized, packed binary shards with the same `LITPKDS`
format used by TinyLLaMA. Shard filenames use prefixes to define subsets
for training, for example `train_parquet_*`.

Expected folder layout:

```
/path/to/packed_data/
  train_parquet_0_0000000000.bin
  train_parquet_0_0000000001.bin
  train_parquet_1_0000000000.bin
```

To build these shards from parquet files in this repo, use
`data_prep.py`. It reads parquet rows in batches, tokenizes a
text column with a Hugging Face tokenizer, appends EOS by default, and packs the
result into fixed-size TinyLLaMA-style `.bin` shards. Work is split across
multiple processes by parquet file.

Example download command for a parquet shard:

```bash
hf download gmongaras/SlimPajama-627B_Reupload --repo-type dataset --include "data/train-00002*" --local-dir ./data
```

Example for running the tokenizer:

```bash
cd ./miniDiffuLLaMA

.venv/bin/python data_prep.py \
  --source-path ../data/data \
  --tokenizer <hf_or_local_tokenizer> \
  --destination-path ../data/packed_data \
  --prefix train_parquet \
  --pattern *.parquet \
  --num-processes 2
```

Useful flags:

- `--text-column`: explicitly choose the parquet text column. If omitted, the
  script tries `text`, `content`, `body`, then falls back to the first string column.
- `--percentage`: process only a fraction of the parquet files.
- `--skip-redpajama-github`: skip rows where `meta.redpajama_set_name == "RedPajamaGithub"`.
- `--write-remainder`: write the final partial shard. By default it is dropped,
  matching TinyLLaMA behavior.
- `--no-eos`: disable automatic EOS appending.

## Train (DDP with Accelerate)

Single node, 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --num_processes 4 \
  train.py \
  --model <hf_or_local_model> \
  --packed-data-dir /path/to/packed_data \
  --seq-length 2048 \
  --batch-size 1 \
  --gradient-accumulate-every 8 \
  --max-train-steps 20000 \
  --learning-rate 1.5e-5 \
  --output-dir /path/to/runs/mini_diffullama \
  --checkpoint-dir /path/to/runs/mini_diffullama/checkpoints \
  --save-every 500 \
  --eval-every 500 \
  --packed-prefixes train_parquet \
  --eval-tasks hellaswag,winogrande,piqa,siqa \
  --wandb-project miniDiffuLLaMA
```

Attention-mask mode notes:

- If you want properly interpolated attention between causal and full attention, use `--attn-impl sdpa`. In this mode the training code applies the custom 4D annealed mask.
- If you want to use FlashAttention 2, use `--attn-impl flash_attention_2`. In this mode the training code skips the annealed 4D mask and switches the model to immediate non-causal attention.

Example with interpolated attention via SDPA:

```bash
accelerate launch --num_processes 1 train.py   --model <hf_or_local_model>   --packed-data-dir /path/to/packed_data   --packed-prefixes train_parquet   --attn-impl sdpa
```

Example with FlashAttention 2 and immediate non-causal attention:

```bash
accelerate launch --num_processes 1 train.py   --model <hf_or_local_model>   --packed-data-dir /path/to/packed_data   --packed-prefixes train_parquet   --attn-impl flash_attention_2
```

If you want to control dataset mixing, pass prefixes and weights:

```bash
--packed-prefixes train_parquet \
--packed-weights 1.0
```

Additional training outputs:

- `--checkpoint-dir`: periodic checkpoints are written here as `step_00000500/`, `step_00001000/`, etc. If omitted, checkpoints default to `<output-dir>/checkpoints`.
- `--save-every`: checkpoint cadence in optimizer steps. `0` disables periodic checkpoints.
- `--eval-every`: evaluation cadence in optimizer steps. `0` disables scheduled evals.
- `--eval-tasks`: choose which evals to run during scheduled evaluation. The default is `hellaswag,winogrande,piqa,siqa`.
- `--eval-history-path`: JSONL file where scheduled eval metrics are appended. If omitted, defaults to `<output-dir>/eval_history.jsonl`.

Example output layout with `--output-dir /path/to/runs/mini_diffullama`:

```text
/path/to/runs/mini_diffullama/
  config.json
  generation_config.json
  model.safetensors or pytorch_model.bin
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  checkpoints/
    step_00000500/
    step_00001000/
  eval_history.jsonl
```

In other words:

- `output_dir` holds the final exported model and tokenizer at the end of training.
- `output_dir/checkpoints/` is the default periodic checkpoint location if `--checkpoint-dir` is not set.
- `output_dir/eval_history.jsonl` is the default scheduled-eval log if `--eval-history-path` is not set.

### W&B reminder

`wandb` is now a project dependency, but you still need to authenticate before logging:

```bash
source .venv/bin/activate
wandb login
```

You can also set the API key non-interactively:

```bash
export WANDB_API_KEY=<your_api_key>
```

Useful W&B flags:

- `--wandb-project`: required to enable W&B logging.
- `--wandb-entity`: optional team or user namespace.
- `--wandb-run-name`: optional custom run name.
- `--wandb-dir`: optional local directory for W&B metadata.

What gets logged:

- `train/loss` every optimizer step.
- `eval/*` metrics every `--eval-every` steps.

## Eval (multiple-choice + generation)

```bash
python eval.py \
  --model ./output/mini_diffullama \
  --diffusion-steps 64 \
  --eval-tasks hellaswag,winogrande,piqa,siqa

# Lambada (uses local file)
python eval.py \
  --model ./output/mini_diffullama \
  --eval-tasks lambada \
  --lambada-path eval_data/lambada_test_plain_text.txt

# Poem reverse (download into miniDiffuLLaMA/eval_data/)
python eval.py \
  --model ./output/mini_diffullama \
  --eval-tasks poem_reverse \
  --poem-path eval_data/poem_data.json \
  --poem-direction ftb \
  --gen-length 28
```

To print a checkpoint-by-metric table from the eval history JSONL:

```bash
python eval.py \
  --history-paths /path/to/runs/mini_diffullama/eval_history.jsonl
```

This prints a table with rows as eval metrics and columns as checkpoints.

Notes:
- `train.py` uses a diffusion masking objective with annealed attention.
- `eval.py` uses denoising-loss scoring for multiple-choice tasks, and generation-based evaluation for tasks like Lambada and poem reverse.

## Packed dataset mechanics (how data is streamed)

`miniDiffuLLaMA/data.py` implements a TinyLLaMA-style packed dataset loader.
Each `.bin` shard is a contiguous stream of token IDs plus a small header. The
loader does **not** read all shards into RAM; instead it memmaps shards and
streams fixed-length blocks.

Key pieces:

- **`PackedDataset`**: an `IterableDataset` that shards files across DDP ranks
  and dataloader workers. Each worker sees a disjoint subset of shard files.
- **`PackedDatasetIterator`**: the core streaming iterator.
  - Loads `n_chunks` shard files at a time with `np.memmap`.
  - Computes how many fixed-length blocks fit in those chunks.
  - Shuffles block indices **within the loaded buffer** (fast, memory-bounded).
  - Yields one block of length `block_size` per `__next__`.
  - When the buffer is exhausted, it loads the next `n_chunks` files.
- **`CombinedDataset`**: mixes multiple packed streams by weighted sampling on
  each `next()` call.

This gives you:
1. **No padding waste** (all blocks are full).
2. **Low RAM usage** (memmap + small buffer).
3. **Scalable sharding** across multiple GPUs/workers.

## Sources and inspiration

This mini pipeline is distilled from the following sources in this repo and related projects:

- **DiffuLLaMA**: diffusion objective, annealed attention, and training flow.
  See `DiffuLLaMA-training/train.py` and `DiffuLLaMA-training/README.md`.
- **LLaMA-Factory (DiffuLLaMA fork)**: DDM training setup and evaluation approach.
  See `LLaMA-Factory/src/llamafactory/train/ddm/`.
- **TinyLLaMA**: packed binary dataset format that this repo reads and writes.
