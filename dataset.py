from torch.utils.data import Dataset
from utils import board_to_tensor
import chess
import torch

class ChessDataset(Dataset):
    """PyTorch Dataset for chess position and move data."""

    def __init__(self, data_file):
        """
        Initialize the chess dataset from a saved data file.

        Args:
            data_file: Path to the .pt file containing (FEN, move_id, result) tuples.
        """
        print("Loading data into RAM...")
        self.data = torch.load(data_file)
        print(f"Loaded {len(self.data)} training examples.")
        
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
        fen, move_id, result = self.data[idx]

        # 1. Convert FEN to Board Tensor
        board = chess.Board(fen)
        x_tensor = board_to_tensor(board)
        
        # 2. Prepare Targets
        # Policy Target: The exact move index (Class Label)
        # Value Target: Who won the game? (Float)
        y_policy = torch.tensor(move_id, dtype=torch.long)
        y_value = torch.tensor(result, dtype=torch.float32)
        
        return x_tensor, y_policy, y_value