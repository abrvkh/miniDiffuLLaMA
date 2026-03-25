import glob
import random
import struct
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}

HDR_MAGIC = b"LITPKDS"
HDR_SIZE = 24  # bytes


class PackedDataset(IterableDataset):
    """
    Streams fixed-length token blocks from TinyLLaMA-style packed .bin shards.

    Each shard is a contiguous stream of token IDs with a small header. We memmap
    shards and yield `block_size` token blocks to avoid loading the full dataset
    into RAM.
    """

    def __init__(
        self,
        filenames: List[str],
        n_chunks: int,
        block_size: int,
        seed: int = 12345,
        shuffle: bool = True,
        wrap: bool = False,
        num_processes: int = 1,
        process_rank: int = 0,
    ) -> None:
        self._filenames = filenames
        self._n_chunks = n_chunks
        self._block_size = block_size
        self._seed = seed
        self._shuffle = shuffle
        self._wrap = wrap
        self._num_processes = num_processes
        self._process_rank = process_rank

    def __iter__(self) -> Iterable[torch.Tensor]:
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        max_num_files = len(self._filenames) // num_shards * num_shards
        filenames = self._filenames[shard_id:max_num_files:num_shards]

        return PackedDatasetIterator(
            filenames=filenames,
            n_chunks=self._n_chunks,
            block_size=self._block_size,
            seed=self._seed,
            shuffle=self._shuffle,
            wrap=self._wrap,
        )


class PackedDatasetIterator:
    def __init__(self, filenames, n_chunks, block_size, seed, shuffle, wrap):
        self._seed = seed
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed) if shuffle else None
        self._wrap = wrap

        self._filenames = filenames
        self._file_idx = 0
        self._n_chunks = n_chunks
        self._dtype = None
        self._block_size = block_size
        self._n_blocks = None
        self._mmaps = []
        self._buffers = []
        self._block_idxs = []
        self._curr_idx = 0

        self._load_n_chunks()

    def _read_header(self, path):
        with open(path, "rb") as f:
            magic = f.read(len(HDR_MAGIC))
            if magic != HDR_MAGIC:
                raise ValueError("File doesn't match expected packed format.")
            (version,) = struct.unpack("<Q", f.read(8))
            if version != 1:
                raise ValueError(f"Unsupported packed version: {version}")
            (dtype_code,) = struct.unpack("<B", f.read(1))
            dtype = DTYPES[dtype_code]
            (chunk_size,) = struct.unpack("<Q", f.read(8))
        return dtype, chunk_size

    def _close_mmaps(self):
        for mmap in self._mmaps:
            mmap._mmap.close()

    def _load_n_chunks(self):
        self._close_mmaps()
        self._mmaps = []
        self._buffers = []

        if self._n_chunks > len(self._filenames[self._file_idx :]):
            self._file_idx = 0

        for i in range(self._n_chunks):
            filename = self._filenames[self._file_idx + i]
            if self._dtype is None:
                self._dtype, self._chunk_size = self._read_header(filename)
                self._n_blocks = self._chunk_size // self._block_size
            mmap = np.memmap(filename, mode="r", order="C", offset=HDR_SIZE)
            self._mmaps.append(mmap)
            self._buffers.append(memoryview(mmap))

        self._file_idx += self._n_chunks
        n_all_blocks = self._n_chunks * self._n_blocks

        self._block_idxs = self._rng.permutation(n_all_blocks) if self._shuffle else range(n_all_blocks)
        self._curr_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._curr_idx >= len(self._block_idxs):
            self._load_n_chunks()
        block_idx = self._block_idxs[self._curr_idx]
        chunk_id = block_idx // self._n_blocks
        buffer = self._buffers[chunk_id]
        elem_id = (block_idx % self._n_blocks) * self._block_size
        offset = np.dtype(self._dtype).itemsize * elem_id
        arr = np.frombuffer(buffer, dtype=self._dtype, count=self._block_size, offset=offset)
        self._curr_idx += 1
        return torch.from_numpy(arr.astype(np.int64))


class CombinedDataset(IterableDataset):
    """
    Mixes multiple PackedDataset streams by weighted sampling.
    """

    def __init__(self, datasets: List[IterableDataset], seed: int, weights: Optional[List[float]] = None):
        self._seed = seed
        self._datasets = datasets
        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)
        self._weights = weights

    def __iter__(self):
        return CombinedDatasetIterator(self._datasets, self._seed, self._weights)


class CombinedDatasetIterator:
    def __init__(self, datasets, seed, weights):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        return next(dataset)


def build_packed_dataloader(
    packed_dir: Path,
    prefixes: Optional[List[str]],
    weights: Optional[List[float]],
    batch_size: int,
    block_size: int,
    num_processes: int,
    process_rank: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """
    Loads TinyLLaMA-style packed binary shards from `packed_dir`.
    Each shard is a .bin file produced by TinyLLaMA preprocessing, containing a
    contiguous stream of token IDs. The shard filename prefix (e.g., "train_parquet")
    selects which subset to read. We memmap the shards and yield fixed-length
    blocks (`block_size`) for training.
    """
    datasets = []
    if not prefixes:
        prefixes = ["train_parquet"]
    if not weights:
        weights = [1.0 / len(prefixes)] * len(prefixes)
    if len(weights) != len(prefixes):
        raise ValueError(f"weights length {len(weights)} must match prefixes length {len(prefixes)}")

    for prefix in prefixes:
        filenames = sorted(glob.glob(str(packed_dir / f"{prefix}*.bin")))
        if not filenames:
            raise RuntimeError(
                f"No packed .bin files found at {packed_dir} for prefix {prefix!r}. "
                f"Pass --packed-prefixes explicitly if your shard names use a different prefix."
            )
        random.seed(seed)
        random.shuffle(filenames)
        dataset = PackedDataset(
            filenames,
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + process_rank,
            num_processes=num_processes,
            process_rank=process_rank,
        )
        datasets.append(dataset)

    s = sum(weights)
    weights = [w / s for w in weights]
    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
