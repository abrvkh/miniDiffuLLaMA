import argparse
import glob
import math
import struct
from multiprocessing import Process, cpu_count
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from data import DTYPES, HDR_MAGIC


def dtype_code(dtype: np.dtype) -> int:
    normalized = np.dtype(dtype).type
    for code, candidate in DTYPES.items():
        if candidate == normalized:
            return code
    raise ValueError(f"Unsupported dtype: {dtype}")


def choose_packed_dtype(vocab_size: int) -> np.dtype:
    return np.uint16 if vocab_size < 65500 else np.int32


def choose_sep_token_id(tokenizer) -> int:
    for token_id in (
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ):
        if token_id is not None:
            return int(token_id)
    return 0


class PackedDatasetBuilder:
    def __init__(self, outdir: Path, prefix: str, chunk_size: int, sep_token: int, vocab_size: int) -> None:
        self._dtype = choose_packed_dtype(vocab_size)
        self._counter = 0
        self._chunk_size = chunk_size
        self._outdir = outdir
        self._prefix = prefix
        self._sep_token = sep_token
        self._arr = np.full(self._chunk_size, self._sep_token, dtype=self._dtype)
        self._idx = 0
        self._version = 1
        self._filenames = []

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def add_array(self, arr: np.ndarray) -> None:
        while self._idx + arr.shape[0] > self._chunk_size:
            part_len = self._chunk_size - self._idx
            self._arr[self._idx : self._idx + part_len] = arr[:part_len]
            self._write_chunk()
            arr = arr[part_len:]

        arr_len = arr.shape[0]
        self._arr[self._idx : self._idx + arr_len] = arr
        self._idx += arr_len

    def write_remainder(self) -> None:
        if self._idx > 0:
            self._write_chunk()

    def _write_chunk(self) -> None:
        filename = self._outdir / f"{self._prefix}_{self._counter:010d}.bin"
        with open(filename, "wb") as f:
            f.write(HDR_MAGIC)
            f.write(struct.pack("<Q", self._version))
            f.write(struct.pack("<B", dtype_code(self._dtype)))
            f.write(struct.pack("<Q", self._chunk_size))
            f.write(self._arr.tobytes(order="C"))

        self._counter += 1
        self._arr.fill(self._sep_token)
        self._idx = 0


def resolve_text_column(parquet_file: pq.ParquetFile, requested: Optional[str]) -> str:
    available = parquet_file.schema_arrow.names
    if requested is not None:
        if requested not in available:
            raise KeyError(f"text column {requested!r} not found. Available columns: {available}")
        return requested

    for candidate in ("text", "content", "body"):
        if candidate in available:
            return candidate

    for field in parquet_file.schema_arrow:
        if str(field.type) == "string":
            return field.name

    raise KeyError(f"Could not infer a text column. Available columns: {available}")


def should_skip_row(row: dict, meta_column: str, skip_redpajama_github: bool) -> bool:
    if not skip_redpajama_github:
        return False

    meta = row.get(meta_column)
    if isinstance(meta, dict) and meta.get("redpajama_set_name") == "RedPajamaGithub":
        return True

    dotted_name = f"{meta_column}.redpajama_set_name"
    return row.get(dotted_name) == "RedPajamaGithub"


def iter_texts(
    parquet_path: Path,
    text_column: Optional[str],
    meta_column: str,
    batch_size: int,
    skip_redpajama_github: bool,
) -> Iterable[str]:
    parquet_file = pq.ParquetFile(parquet_path)
    resolved_text_column = resolve_text_column(parquet_file, text_column)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            if should_skip_row(row, meta_column, skip_redpajama_github):
                continue
            text = row.get(resolved_text_column)
            if isinstance(text, str) and text:
                yield text


def encode_text(tokenizer, text: str, append_eos: bool) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if append_eos and tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    return token_ids


def process_subset(
    parquet_files: list[str],
    tokenizer_name_or_path: str,
    destination_path: str,
    prefix: str,
    chunk_size: int,
    text_column: Optional[str],
    meta_column: str,
    batch_size: int,
    append_eos: bool,
    skip_redpajama_github: bool,
    write_remainder: bool,
    process_id: int,
) -> None:
    output_dir = Path(destination_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    builder = PackedDatasetBuilder(
        outdir=output_dir,
        prefix=f"{prefix}_{process_id}",
        chunk_size=chunk_size,
        sep_token=choose_sep_token_id(tokenizer),
        vocab_size=len(tokenizer),
    )

    for parquet_file in parquet_files:
        print(f"[worker {process_id}] processing {parquet_file}")
        for text in iter_texts(
            parquet_path=Path(parquet_file),
            text_column=text_column,
            meta_column=meta_column,
            batch_size=batch_size,
            skip_redpajama_github=skip_redpajama_github,
        ):
            token_ids = encode_text(tokenizer, text, append_eos=append_eos)
            if not token_ids:
                continue
            builder.add_array(np.asarray(token_ids, dtype=builder.dtype))

    if write_remainder:
        builder.write_remainder()


def discover_parquet_files(source_path: Path, pattern: str, percentage: float) -> list[str]:
    parquet_files = sorted(glob.glob(str(source_path / pattern), recursive=True))
    if not parquet_files:
        raise RuntimeError(f"No parquet files matching {pattern!r} found under {source_path}")

    if percentage <= 0 or percentage > 1:
        raise ValueError("percentage must be in the interval (0, 1].")

    limit = max(1, math.floor(len(parquet_files) * percentage))
    return parquet_files[:limit]


def prepare_parquet(
    source_path: Path,
    tokenizer: str,
    destination_path: Path,
    prefix: str = "train_parquet",
    pattern: str = "*.parquet",
    text_column: Optional[str] = None,
    meta_column: str = "meta",
    chunk_size: int = 2049 * 1024,
    batch_size: int = 1024,
    percentage: float = 1.0,
    num_processes: Optional[int] = None,
    append_eos: bool = True,
    skip_redpajama_github: bool = False,
    write_remainder: bool = False,
) -> None:
    parquet_files = discover_parquet_files(source_path, pattern=pattern, percentage=percentage)
    worker_count = num_processes or cpu_count()
    worker_count = max(1, min(worker_count, len(parquet_files)))
    subsets = [list(chunk) for chunk in np.array_split(parquet_files, worker_count) if len(chunk) > 0]

    processes = []
    for process_id, subset in enumerate(subsets):
        process = Process(
            target=process_subset,
            args=(
                subset,
                tokenizer,
                str(destination_path),
                prefix,
                chunk_size,
                text_column,
                meta_column,
                batch_size,
                append_eos,
                skip_redpajama_github,
                write_remainder,
                process_id,
            ),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {process.exitcode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack parquet text into TinyLLaMA-style .bin shards.")
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--destination-path", type=Path, required=True)
    parser.add_argument("--prefix", type=str, default="train_parquet")
    parser.add_argument("--pattern", type=str, default="*.parquet")
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--meta-column", type=str, default="meta")
    parser.add_argument("--chunk-size", type=int, default=2049 * 1024)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--percentage", type=float, default=1.0)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--skip-redpajama-github", action="store_true")
    parser.add_argument("--no-eos", action="store_true")
    parser.add_argument("--write-remainder", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    prepare_parquet(
        source_path=args.source_path,
        tokenizer=args.tokenizer,
        destination_path=args.destination_path,
        prefix=args.prefix,
        pattern=args.pattern,
        text_column=args.text_column,
        meta_column=args.meta_column,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        percentage=args.percentage,
        num_processes=args.num_processes,
        append_eos=not args.no_eos,
        skip_redpajama_github=args.skip_redpajama_github,
        write_remainder=args.write_remainder,
    )
