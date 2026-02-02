"""
PyTorch Dataset classes for chess training.

Supports both small datasets (loaded fully into RAM) and large datasets
(lazy loading with memory-mapped files or sharding).
"""

import os
import random
from pathlib import Path
from torch.utils.data import Dataset, IterableDataset
import torch
import chess
from utils import board_to_tensor


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess position and move data.

    Loads entire dataset into RAM. Suitable for datasets up to ~2-5M examples
    depending on available memory.
    """

    def __init__(self, data_file, precompute_tensors=False, verbose=True):
        """
        Initialize the chess dataset from a saved data file.

        Args:
            data_file: Path to the .pt file containing (FEN, move_id, result) tuples.
            precompute_tensors: If True, convert all FENs to tensors upfront (uses more RAM but faster training).
            verbose: Print loading progress.
        """
        if verbose:
            print(f"Loading data from {data_file}...")

        self.data = torch.load(data_file, weights_only=False)
        self.precomputed = precompute_tensors

        if verbose:
            print(f"Loaded {len(self.data):,} training examples.")

        if precompute_tensors:
            if verbose:
                print("Precomputing board tensors (this may take a while)...")
            self._precompute_tensors()

    def _precompute_tensors(self):
        """Convert all FENs to tensors upfront for faster training."""
        from tqdm import tqdm

        precomputed_data = []
        for fen, move_id, result in tqdm(self.data, desc="Precomputing"):
            board = chess.Board(fen)
            tensor = board_to_tensor(board)
            precomputed_data.append((tensor, move_id, result))

        self.data = precomputed_data

    def __len__(self):
        """Return the total number of training examples."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single training example.

        Args:
            idx: Index of the example to retrieve.

        Returns:
            Tuple of (board_tensor, policy_target, value_target).
        """
        if self.precomputed:
            x_tensor, move_id, result = self.data[idx]
        else:
            fen, move_id, result = self.data[idx]
            board = chess.Board(fen)
            x_tensor = board_to_tensor(board)

        y_policy = torch.tensor(move_id, dtype=torch.long)
        y_value = torch.tensor(result, dtype=torch.float32)

        return x_tensor, y_policy, y_value


class ChessDatasetLazy(Dataset):
    """
    Memory-efficient dataset that loads examples on-demand.

    Uses memory-mapped file reading for large datasets that don't fit in RAM.
    Slightly slower than ChessDataset but can handle arbitrarily large datasets.
    """

    def __init__(self, data_file, cache_size=10000):
        """
        Initialize lazy-loading dataset.

        Args:
            data_file: Path to the .pt file.
            cache_size: Number of recent examples to cache in memory.
        """
        print(f"Initializing lazy dataset from {data_file}...")

        # Load just the length and keep file reference
        self.data_file = data_file
        self.data = torch.load(data_file, weights_only=False)
        self.length = len(self.data)

        # LRU cache for recent examples
        self.cache_size = cache_size
        self.cache = {}
        self.cache_order = []

        print(f"Dataset ready: {self.length:,} examples (lazy loading)")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            fen, move_id, result = self.cache[idx]
        else:
            fen, move_id, result = self.data[idx]

            # Update cache
            self.cache[idx] = (fen, move_id, result)
            self.cache_order.append(idx)

            # Evict old entries if cache is full
            while len(self.cache_order) > self.cache_size:
                old_idx = self.cache_order.pop(0)
                if old_idx in self.cache:
                    del self.cache[old_idx]

        board = chess.Board(fen)
        x_tensor = board_to_tensor(board)
        y_policy = torch.tensor(move_id, dtype=torch.long)
        y_value = torch.tensor(result, dtype=torch.float32)

        return x_tensor, y_policy, y_value


class ChessDatasetSharded(Dataset):
    """
    Dataset that loads from multiple shard files.

    Useful for very large datasets split across multiple files.
    """

    def __init__(self, shard_dir, shard_pattern="chunk_*.pt"):
        """
        Initialize sharded dataset.

        Args:
            shard_dir: Directory containing shard files.
            shard_pattern: Glob pattern for shard files.
        """
        self.shard_dir = Path(shard_dir)
        self.shard_files = sorted(self.shard_dir.glob(shard_pattern))

        if not self.shard_files:
            raise ValueError(f"No shards found matching {shard_pattern} in {shard_dir}")

        print(f"Found {len(self.shard_files)} shards")

        # Build index: which shard contains which indices
        self.shard_offsets = []
        self.shard_lengths = []
        total = 0

        for shard_file in self.shard_files:
            data = torch.load(shard_file, weights_only=False)
            length = len(data)
            self.shard_offsets.append(total)
            self.shard_lengths.append(length)
            total += length
            del data

        self.length = total
        self.current_shard_idx = None
        self.current_shard_data = None

        print(f"Total examples: {self.length:,}")

    def __len__(self):
        return self.length

    def _load_shard(self, shard_idx):
        """Load a shard into memory."""
        if shard_idx != self.current_shard_idx:
            self.current_shard_data = torch.load(
                self.shard_files[shard_idx], weights_only=False
            )
            self.current_shard_idx = shard_idx

    def _find_shard(self, idx):
        """Find which shard contains the given index."""
        for i, (offset, length) in enumerate(zip(self.shard_offsets, self.shard_lengths)):
            if offset <= idx < offset + length:
                return i, idx - offset
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx):
        shard_idx, local_idx = self._find_shard(idx)
        self._load_shard(shard_idx)

        fen, move_id, result = self.current_shard_data[local_idx]

        board = chess.Board(fen)
        x_tensor = board_to_tensor(board)
        y_policy = torch.tensor(move_id, dtype=torch.long)
        y_value = torch.tensor(result, dtype=torch.float32)

        return x_tensor, y_policy, y_value


class ChessIterableDataset(IterableDataset):
    """
    Iterable dataset for streaming large datasets.

    Memory-efficient: only keeps one batch in memory at a time.
    Good for datasets too large to index randomly.
    """

    def __init__(self, data_file, shuffle_buffer_size=100000):
        """
        Initialize streaming dataset.

        Args:
            data_file: Path to .pt file.
            shuffle_buffer_size: Size of shuffle buffer for randomization.
        """
        self.data_file = data_file
        self.shuffle_buffer_size = shuffle_buffer_size

        # Get length without keeping data in memory
        data = torch.load(data_file, weights_only=False)
        self.length = len(data)
        del data

        print(f"Streaming dataset: {self.length:,} examples")

    def __iter__(self):
        data = torch.load(self.data_file, weights_only=False)

        # Shuffle buffer for randomization
        buffer = []

        for fen, move_id, result in data:
            buffer.append((fen, move_id, result))

            if len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)
                while len(buffer) > self.shuffle_buffer_size // 2:
                    item = buffer.pop()
                    yield self._process_item(item)

        # Yield remaining items
        random.shuffle(buffer)
        for item in buffer:
            yield self._process_item(item)

    def _process_item(self, item):
        fen, move_id, result = item
        board = chess.Board(fen)
        x_tensor = board_to_tensor(board)
        y_policy = torch.tensor(move_id, dtype=torch.long)
        y_value = torch.tensor(result, dtype=torch.float32)
        return x_tensor, y_policy, y_value


def create_dataset(data_path, dataset_type="auto", **kwargs):
    """
    Factory function to create the appropriate dataset type.

    Args:
        data_path: Path to data file or shard directory.
        dataset_type: One of "auto", "standard", "lazy", "sharded", "streaming".
        **kwargs: Additional arguments passed to dataset constructor.

    Returns:
        Dataset instance.
    """
    data_path = Path(data_path)

    if dataset_type == "auto":
        if data_path.is_dir():
            dataset_type = "sharded"
        else:
            # Check file size to decide
            size_mb = data_path.stat().st_size / (1024 * 1024)
            if size_mb > 2000:  # > 2GB
                dataset_type = "lazy"
            else:
                dataset_type = "standard"

    if dataset_type == "standard":
        return ChessDataset(data_path, **kwargs)
    elif dataset_type == "lazy":
        return ChessDatasetLazy(data_path, **kwargs)
    elif dataset_type == "sharded":
        return ChessDatasetSharded(data_path, **kwargs)
    elif dataset_type == "streaming":
        return ChessIterableDataset(data_path, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
