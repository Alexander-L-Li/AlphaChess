"""
Modal-compatible training script for AlphaZero-style chess training.

Refactored from alpha_train.py with:
- Device-agnostic code (CUDA > MPS > CPU)
- Configurable paths for Modal Volume mounting
- Checkpoint resumption support
- Parameterized hyperparameters
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import chess
from collections import deque
import random
import gc
import time
from pathlib import Path
from tqdm import tqdm

from mcts import BatchedMCTS
from utils import ActionMapper, board_to_tensor
from nn import ChessNet


def get_device():
    """Detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_device_cache(device):
    """Clear device-specific cache and run garbage collection."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()


def execute_episode(mcts, mapper, mcts_sims):
    """
    Execute one full game of self-play to generate training data.

    Plays a complete game from the starting position using MCTS to select moves.
    For the first 30 moves, moves are sampled proportionally to visit counts
    to encourage exploration. After that, the most-visited move is chosen.

    Args:
        mcts: BatchedMCTS instance used for move selection.
        mapper: ActionMapper for converting moves to policy vector indices.
        mcts_sims: Number of MCTS simulations per move.

    Returns:
        List of training examples as (board_tensor, policy_vector, value) tuples.
    """
    examples = []
    board = chess.Board()
    mcts_root = None

    while not board.is_game_over():
        root = mcts.run_self_play_simulation(board, num_simulations=mcts_sims, root=mcts_root)

        visit_counts = [child.visit_count for child in root.children.values()]
        moves = [move for move in root.children.keys()]
        sum_visits = sum(visit_counts)
        pi = [v / sum_visits for v in visit_counts]

        examples.append([board.copy(), moves, pi, None])

        # Temperature-based move selection
        if len(board.move_stack) < 30:
            chosen_move = np.random.choice(moves, p=pi)
        else:
            chosen_move = moves[np.argmax(visit_counts)]

        board.push(chosen_move)

        # Subtree reuse
        if chosen_move in root.children:
            mcts_root = root.children[chosen_move]
            mcts_root.parent = None
        else:
            mcts_root = None

    # Process game outcome
    outcome = board.outcome()
    if outcome.winner is None:
        result = 0
    else:
        result = 1 if outcome.winner == chess.WHITE else -1

    processed_examples = []
    for state, moves, pi, _ in examples:
        player_result = result if state.turn == chess.WHITE else -result

        pi_vector = np.zeros(mapper.vocab_size, dtype=np.float32)
        for i, move in enumerate(moves):
            idx = mapper.encode(move)
            if idx is not None:
                pi_vector[idx] = pi[i]

        processed_examples.append((board_to_tensor(state), pi_vector, player_result))

    return processed_examples


def train(model, replay_buffer, device, epochs, batch_size):
    """
    Train the neural network on data from the replay buffer.

    Args:
        model: The ChessNet model.
        replay_buffer: Deque containing training examples.
        device: Torch device.
        epochs: Number of training epochs.
        batch_size: Training batch size.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    mse = nn.MSELoss()

    model.train()

    if len(replay_buffer) < batch_size:
        print(f"  Buffer too small ({len(replay_buffer)} < {batch_size}), skipping training")
        return

    buffer_list = list(replay_buffer)
    batch_count = min(len(buffer_list) // batch_size, 500)

    print(f"  Training for {epochs} epochs on {len(buffer_list)} examples ({batch_count} batches/epoch)...")

    for epoch in range(epochs):
        total_loss = 0
        total_loss_v = 0
        total_loss_p = 0

        for _ in range(batch_count):
            batch = random.sample(buffer_list, batch_size)

            state_tensor = torch.stack([b[0] for b in batch]).to(device)
            policy_target = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32).to(device)
            value_target = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(device)

            p_pred, v_pred = model(state_tensor)

            loss_v = mse(v_pred.squeeze(), value_target)
            loss_p = -torch.sum(policy_target * torch.log_softmax(p_pred, dim=1)) / batch_size
            loss = loss_v + loss_p

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()

        avg_loss = total_loss / batch_count
        avg_loss_v = total_loss_v / batch_count
        avg_loss_p = total_loss_p / batch_count
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (v={avg_loss_v:.4f}, p={avg_loss_p:.4f})")


def save_checkpoint(checkpoint_path, iteration, model, replay_buffer):
    """Save training checkpoint for resumption."""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'replay_buffer': list(replay_buffer)
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: iteration {iteration}")


def load_checkpoint(checkpoint_path, model, max_buffer_size):
    """Load training checkpoint if it exists."""
    if not Path(checkpoint_path).exists():
        return 0, deque(maxlen=max_buffer_size)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    replay_buffer = deque(checkpoint['replay_buffer'], maxlen=max_buffer_size)
    start_iteration = checkpoint['iteration']

    print(f"Resumed from iteration {start_iteration}, buffer size: {len(replay_buffer)}")
    return start_iteration, replay_buffer


def run_training(
    # Paths
    model_input_path: str = "supervised_chess_model.pth",
    model_output_path: str = "modal_model.pth",
    checkpoint_path: str = "rl_checkpoint.pth",
    # Hyperparameters
    iterations: int = 10,
    games_per_iter: int = 1000,
    mcts_sims: int = 800,
    batch_size: int = 256,
    epochs: int = 10,
    max_buffer_size: int = 100000,
    mcts_batch_size: int = 32,
    cache_size: int = 50000,
    # Runtime limit (for Modal's 24hr timeout)
    max_runtime_hours: float = 23.0,
):
    """
    Main training loop for AlphaZero-style reinforcement learning.

    Args:
        model_input_path: Path to pre-trained model weights.
        model_output_path: Path to save trained model.
        checkpoint_path: Path for checkpoint file (for resumption).
        iterations: Number of self-play/training iterations.
        games_per_iter: Games to play per iteration.
        mcts_sims: MCTS simulations per move.
        batch_size: Training batch size.
        epochs: Training epochs per iteration.
        max_buffer_size: Maximum replay buffer size.
        mcts_batch_size: Batch size for MCTS neural net evaluation.
        cache_size: Transposition table size.
    """
    device = get_device()
    print(f"RL Training on {device}")
    print(f"Config: {games_per_iter} games/iter, {mcts_sims} sims, batch={mcts_batch_size}")

    mapper = ActionMapper()
    model = ChessNet(action_size=mapper.vocab_size).to(device)

    # Load pre-trained weights
    if Path(model_input_path).exists():
        model.load_state_dict(torch.load(model_input_path, map_location=device))
        print(f"Loaded pre-trained model from {model_input_path}")
    else:
        print(f"Warning: No pre-trained model found at {model_input_path}, starting from scratch")

    # Try to resume from checkpoint
    start_iteration, replay_buffer = load_checkpoint(checkpoint_path, model, max_buffer_size)

    mcts = BatchedMCTS(model, device, mapper, batch_size=mcts_batch_size, cache_size=cache_size)

    # Track runtime for graceful exit before timeout
    start_time = time.time()
    max_runtime_seconds = max_runtime_hours * 3600

    for i in range(start_iteration, iterations):
        # Check if we're approaching the time limit
        elapsed = time.time() - start_time
        if elapsed > max_runtime_seconds:
            print(f"\n⏰ Approaching time limit ({max_runtime_hours}h). Saving and exiting gracefully.")
            print(f"   Completed {i}/{iterations} iterations. Restart to continue from checkpoint.")
            break
        print(f"\n{'='*50}")
        print(f"--- Iteration {i+1}/{iterations} ---")
        print(f"{'='*50}")

        # Phase 1: Self-Play
        print("Self-Playing...")
        new_examples = []
        for g in tqdm(range(games_per_iter), desc="Games"):
            game_data = execute_episode(mcts, mapper, mcts_sims)
            new_examples.extend(game_data)
            mcts.clear_cache()

        replay_buffer.extend(new_examples)
        print(f"Buffer size: {len(replay_buffer)} | New examples: {len(new_examples)}")

        # Phase 2: Training
        print("Training Network...")
        train(model, replay_buffer, device, epochs, batch_size)

        # Memory cleanup
        clear_device_cache(device)

        # Save checkpoint
        save_checkpoint(checkpoint_path, i + 1, model, replay_buffer)

        # Save current model
        torch.save(model.state_dict(), model_output_path)
        print(f"Model saved to {model_output_path}")

    print("\nTraining complete!")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaZero Chess Training")
    parser.add_argument("--model-input", default="supervised_chess_model.pth", help="Input model path")
    parser.add_argument("--model-output", default="rl_chess_model_latest.pth", help="Output model path")
    parser.add_argument("--checkpoint", default="rl_checkpoint.pth", help="Checkpoint path")
    parser.add_argument("--iterations", type=int, default=10, help="Training iterations")
    parser.add_argument("--games-per-iter", type=int, default=50, help="Games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=200, help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=256, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per iteration")
    parser.add_argument("--max-buffer-size", type=int, default=100000, help="Max replay buffer size")
    parser.add_argument("--mcts-batch-size", type=int, default=32, help="MCTS batch size")
    parser.add_argument("--cache-size", type=int, default=50000, help="Transposition table size")

    args = parser.parse_args()

    run_training(
        model_input_path=args.model_input,
        model_output_path=args.model_output,
        checkpoint_path=args.checkpoint,
        iterations=args.iterations,
        games_per_iter=args.games_per_iter,
        mcts_sims=args.mcts_sims,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_buffer_size=args.max_buffer_size,
        mcts_batch_size=args.mcts_batch_size,
        cache_size=args.cache_size,
    )
