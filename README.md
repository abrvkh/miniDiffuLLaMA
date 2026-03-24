# miniDiffuLLaMA

Minimal diffusion-LM training and eval using TinyLLaMA-style packed datasets.

## Setup (uv)

This repo uses standard Python deps. With `uv`, you can create a venv and sync:

```bash
uv venv
source .venv/bin/activate

uv sync
```

If you need Flash Attention 2 (optional, GPU only):

```bash
uv sync --extra flash-attn
```

## Data (packed .bin shards)

This pipeline expects pre-tokenized, packed binary shards produced by the
TinyLLaMA preprocessing scripts. Each shard is a contiguous stream of token IDs
with a small header. Shard filenames use prefixes to define subsets (e.g.,
`train_slim_*`, `train_star_*`).

Expected folder layout:

```
/path/to/packed_data/
  train_slim_0000000000.bin
  train_slim_0000000001.bin
  train_star_0000000000.bin
  train_star_0000000001.bin
```

To build these shards, follow TinyLLaMA preprocessing. Example (from their docs):

```bash
# Download datasets (large: SlimPajama ~893GB, Starcoderdata ~290GB)
cd /path/to/dataset
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
git clone https://huggingface.co/datasets/bigcode/starcoderdata

# Tokenize + pack into TinyLLaMA-style .bin shards
python scripts/prepare_starcoder.py \
  --source_path /path/to/starcoderdata/ \
  --tokenizer_path data/llama \
  --destination_path data/slim_star_combined \
  --split train \
  --percentage 1.0

python scripts/prepare_slimpajama.py \
  --source_path /path/to/SlimPajama \
  --tokenizer_path data/llama \
  --destination_path data/slim_star_combined \
  --split validation \
  --percentage 1.0

python scripts/prepare_slimpajama.py \
  --source_path /path/to/SlimPajama \
  --tokenizer_path data/llama \
  --destination_path data/slim_star_combined \
  --split train \
  --percentage 1.0
```

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
  --output-dir ./output/mini_diffullama
```

If you want to control dataset mixing, pass prefixes and weights:

```bash
--packed-prefixes train_slim,train_star \
--packed-weights 0.7,0.3
```

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

Notes:
- `train.py` uses a diffusion masking objective with annealed attention.
- `eval.py` scores options by denoising loss; lower loss = better.

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
- **`CombinedDataset`**: mixes multiple packed streams (e.g., `train_slim` and
  `train_star`) by weighted sampling on each `next()` call.

This gives you:
1) **No padding waste** (all blocks are full).
2) **Low RAM usage** (memmap + small buffer).
3) **Scalable sharding** across multiple GPUs/workers.

## Sources and inspiration

This mini pipeline is distilled from the following sources in this repo and related projects:

- **DiffuLLaMA**: diffusion objective, annealed attention, and training flow.
  See `DiffuLLaMA-training/train.py` and `DiffuLLaMA-training/README.md`.
- **LLaMA-Factory (DiffuLLaMA fork)**: DDM training setup and evaluation approach.
  See `LLaMA-Factory/src/llamafactory/train/ddm/`.
- **TinyLLaMA**: packed binary dataset format and preprocessing pipeline.
  Our packed dataloader follows the TinyLLaMA PRETRAIN instructions and
  the `PackedDataset` design in `DiffuLLaMA-training/packed_dataset.py`
  (which is explicitly noted as inspired by Fairseq/Megatron in its header).
