"""
Modal-compatible training script for AlphaZero-style chess training.

Refactored from alpha_train.py with:
- 119-plane AlphaZero-style input representation
- Larger model architecture (10 res blocks, 128 channels, SE blocks)
- Persistent optimizer with LR schedule
- Arena evaluation for model comparison
- Device-agnostic code (CUDA > MPS > CPU)
- Checkpoint resumption support
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
from utils import ActionMapper, board_to_tensor, INPUT_PLANES
from nn import ChessNet, DEFAULT_RES_BLOCKS, DEFAULT_CHANNELS


# --- Default Hyperparameters ---
DEFAULT_CONFIG = {
    # Training iterations
    "iterations": 100,
    "games_per_iter": 500,
    "mcts_sims": 400,
    "batch_size": 256,
    "epochs": 50,
    "max_buffer_size": 500000,
    "mcts_batch_size": 32,
    "cache_size": 100000,

    # Model architecture
    "num_res_blocks": DEFAULT_RES_BLOCKS,
    "num_channels": DEFAULT_CHANNELS,
    "use_se": True,

    # Learning rate schedule
    "initial_lr": 0.01,
    "lr_drops": {30: 0.002, 60: 0.0002, 80: 0.00002},

    # Arena settings
    "arena_games": 100,
    "win_threshold": 0.57,
    "arena_interval": 1,

    # Temperature settings
    "temp_threshold": 30,
    "temp_final": 0.1,

    # Gradient clipping
    "max_grad_norm": 1.0,

    # Runtime limit
    "max_runtime_hours": 23.0,
}


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


def execute_episode(mcts, mapper, mcts_sims, temp_threshold=30, temp_final=0.1):
    """
    Execute one full game of self-play to generate training data.

    Uses temperature-based move selection for exploration/exploitation balance.

    Args:
        mcts: BatchedMCTS instance used for move selection.
        mapper: ActionMapper for converting moves to policy vector indices.
        mcts_sims: Number of MCTS simulations per move.
        temp_threshold: Number of moves to use temperature=1.
        temp_final: Temperature after threshold (near-greedy).

    Returns:
        List of training examples as (board_tensor, policy_vector, value) tuples.
    """
    examples = []
    board = chess.Board()
    mcts_root = None

    while not board.is_game_over():
        root = mcts.run_self_play_simulation(board, num_simulations=mcts_sims, root=mcts_root)

        visit_counts = np.array([child.visit_count for child in root.children.values()])
        moves = list(root.children.keys())
        sum_visits = visit_counts.sum()

        if sum_visits == 0:
            break

        # Temperature-based move selection
        move_number = len(board.move_stack)
        if move_number < temp_threshold:
            pi = visit_counts / sum_visits
            chosen_idx = np.random.choice(len(moves), p=pi)
        else:
            adjusted = visit_counts ** (1.0 / temp_final)
            pi = adjusted / adjusted.sum()
            chosen_idx = np.random.choice(len(moves), p=pi)

        chosen_move = moves[chosen_idx]
        examples.append([board.copy(), moves, visit_counts / sum_visits, None])
        board.push(chosen_move)

        # Subtree reuse
        if chosen_move in root.children:
            mcts_root = root.children[chosen_move]
            mcts_root.parent = None
        else:
            mcts_root = None

    # Process game outcome
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
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


def train(model, optimizer, replay_buffer, device, epochs, batch_size, max_grad_norm=1.0):
    """
    Train the neural network on data from the replay buffer.

    Args:
        model: The ChessNet model.
        optimizer: Persistent optimizer.
        replay_buffer: Deque containing training examples.
        device: Torch device.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        max_grad_norm: Gradient clipping threshold.

    Returns:
        Average loss over training.
    """
    mse = nn.MSELoss()
    model.train()

    if len(replay_buffer) < batch_size:
        print(f"  Buffer too small ({len(replay_buffer)} < {batch_size}), skipping training")
        return 0.0

    buffer_list = list(replay_buffer)
    batch_count = min(len(buffer_list) // batch_size, 1000)

    print(f"  Training for {epochs} epochs on {len(buffer_list)} examples "
          f"({batch_count} batches/epoch, lr={optimizer.param_groups[0]['lr']:.6f})")

    total_epoch_loss = 0

    for epoch in range(epochs):
        total_loss = 0
        total_loss_v = 0
        total_loss_p = 0

        for _ in range(batch_count):
            batch = random.sample(buffer_list, batch_size)

            state_tensor = torch.stack([b[0] for b in batch]).to(device)
            policy_target = torch.tensor(
                np.array([b[1] for b in batch]), dtype=torch.float32
            ).to(device)
            value_target = torch.tensor(
                np.array([b[2] for b in batch]), dtype=torch.float32
            ).to(device)

            p_pred, v_pred = model(state_tensor)

            loss_v = mse(v_pred.squeeze(), value_target)
            loss_p = -torch.sum(policy_target * torch.log_softmax(p_pred, dim=1)) / batch_size
            loss = loss_v + loss_p

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            total_loss_v += loss_v.item()
            total_loss_p += loss_p.item()

        avg_loss = total_loss / batch_count
        avg_loss_v = total_loss_v / batch_count
        avg_loss_p = total_loss_p / batch_count
        total_epoch_loss += avg_loss

        print(f"    Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} "
              f"(v={avg_loss_v:.4f}, p={avg_loss_p:.4f})")

    return total_epoch_loss / epochs


def play_arena_game(model1, model2, device, mapper, mcts_batch_size=32):
    """Play a single game between two models."""
    mcts1 = BatchedMCTS(model1, device, mapper, batch_size=mcts_batch_size)
    mcts2 = BatchedMCTS(model2, device, mapper, batch_size=mcts_batch_size)

    board = chess.Board()

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            mcts = mcts1
        else:
            mcts = mcts2

        root = mcts.search(board, num_simulations=100)
        if not root.children:
            break

        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        board.push(best_move)

        mcts1.clear_cache()
        mcts2.clear_cache()

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def arena_compare(new_model, old_model, device, mapper, num_games=40, mcts_batch_size=32):
    """Compare two models by playing games between them."""
    new_wins = 0
    old_wins = 0
    draws = 0

    for i in range(num_games):
        if i % 2 == 0:
            result = play_arena_game(new_model, old_model, device, mapper, mcts_batch_size)
            if result == 1:
                new_wins += 1
            elif result == -1:
                old_wins += 1
            else:
                draws += 1
        else:
            result = play_arena_game(old_model, new_model, device, mapper, mcts_batch_size)
            if result == 1:
                old_wins += 1
            elif result == -1:
                new_wins += 1
            else:
                draws += 1

        print(f"    Arena game {i+1}/{num_games}: New={new_wins}, Old={old_wins}, Draws={draws}")

    total_played = new_wins + old_wins + draws
    win_rate = (new_wins + 0.5 * draws) / total_played if total_played > 0 else 0.5
    return win_rate


def update_learning_rate(optimizer, iteration, lr_drops):
    """Update learning rate based on iteration count."""
    for threshold, lr in sorted(lr_drops.items()):
        if iteration >= threshold:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def save_checkpoint(checkpoint_path, iteration, model, optimizer, replay_buffer):
    """Save training checkpoint for resumption."""
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'replay_buffer': list(replay_buffer)[-100000:]  # Limit checkpoint size
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: iteration {iteration}")


def load_checkpoint(checkpoint_path, model, optimizer, max_buffer_size, device):
    """Load training checkpoint if it exists and is compatible."""
    if not Path(checkpoint_path).exists():
        return 0, deque(maxlen=max_buffer_size)

    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        replay_buffer = deque(checkpoint.get('replay_buffer', []), maxlen=max_buffer_size)
        start_iteration = checkpoint['iteration']

        print(f"Resumed from iteration {start_iteration}, buffer size: {len(replay_buffer)}")
        return start_iteration, replay_buffer

    except RuntimeError as e:
        if "size mismatch" in str(e) or "Missing key" in str(e):
            print(f"Checkpoint incompatible with current model architecture.")
            print(f"Starting fresh training (old checkpoint will be overwritten).")
            return 0, deque(maxlen=max_buffer_size)
        raise  # Re-raise if it's a different error


def run_training(
    # Paths
    supervised_model_path: str = "supervised_chess_model.pth",
    model_output_path: str = "rl_chess_model_latest.pth",
    best_model_path: str = "rl_chess_model_best.pth",
    checkpoint_path: str = "rl_checkpoint.pth",
    # Config override
    config: dict = None,
):
    """
    Main training loop for AlphaZero-style reinforcement learning.

    Uses a "supervised baseline" approach where RL models must beat the
    original supervised model to be accepted. This prevents catastrophic
    forgetting of supervised knowledge.

    Args:
        supervised_model_path: Path to supervised model (used as baseline).
        model_output_path: Path to save latest model.
        best_model_path: Path to save best RL model that beats baseline.
        checkpoint_path: Path for checkpoint file.
        config: Configuration override dictionary.
    """
    # Merge config with defaults
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    device = get_device()
    print(f"AlphaZero Training on {device}")
    print(f"Config: {cfg['games_per_iter']} games/iter, {cfg['mcts_sims']} sims, "
          f"res_blocks={cfg['num_res_blocks']}, channels={cfg['num_channels']}")

    mapper = ActionMapper()

    # Initialize model with larger architecture
    model = ChessNet(
        action_size=mapper.vocab_size,
        input_channels=INPUT_PLANES,
        num_res_blocks=cfg['num_res_blocks'],
        num_channels=cfg['num_channels'],
        use_se=cfg['use_se']
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Initialize best model for tracking best RL model so far
    best_model = ChessNet(
        action_size=mapper.vocab_size,
        input_channels=INPUT_PLANES,
        num_res_blocks=cfg['num_res_blocks'],
        num_channels=cfg['num_channels'],
        use_se=cfg['use_se']
    ).to(device)

    # Initialize baseline model (supervised model, never changes)
    baseline_model = ChessNet(
        action_size=mapper.vocab_size,
        input_channels=INPUT_PLANES,
        num_res_blocks=cfg['num_res_blocks'],
        num_channels=cfg['num_channels'],
        use_se=cfg['use_se']
    ).to(device)

    # Persistent optimizer (SGD with momentum like AlphaZero)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg['initial_lr'],
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    # Load supervised model as baseline (never changes during training)
    if Path(supervised_model_path).exists():
        try:
            state_dict = torch.load(supervised_model_path, map_location=device)
            baseline_model.load_state_dict(state_dict)
            print(f"Loaded supervised baseline from {supervised_model_path}")
        except Exception as e:
            print(f"ERROR: Could not load supervised baseline: {e}")
            print("Supervised baseline is required for training!")
            raise
    else:
        raise FileNotFoundError(f"Supervised model not found at {supervised_model_path}")

    # Try to resume from checkpoint first
    start_iteration, replay_buffer = load_checkpoint(
        checkpoint_path, model, optimizer, cfg['max_buffer_size'], device
    )

    # If no checkpoint, initialize from supervised model or best RL model
    if start_iteration == 0:
        if Path(best_model_path).exists():
            # Resume from best RL model if it exists
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best RL model from {best_model_path}")
        else:
            # Start from supervised model
            model.load_state_dict(baseline_model.state_dict())
            print(f"Starting from supervised baseline")

    # Copy to best model (best RL model so far)
    if Path(best_model_path).exists() and start_iteration > 0:
        best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        best_model.load_state_dict(model.state_dict())

    mcts = BatchedMCTS(
        model, device, mapper,
        batch_size=cfg['mcts_batch_size'],
        cache_size=cfg['cache_size']
    )

    # Track runtime
    start_time = time.time()
    max_runtime_seconds = cfg['max_runtime_hours'] * 3600

    for iteration in range(start_iteration, cfg['iterations']):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed > max_runtime_seconds:
            print(f"\n⏰ Approaching time limit. Saving and exiting.")
            break

        print(f"\n{'='*60}")
        print(f"--- Iteration {iteration+1}/{cfg['iterations']} ---")
        print(f"{'='*60}")

        # Update learning rate
        update_learning_rate(optimizer, iteration, cfg['lr_drops'])

        # Phase 1: Self-Play
        print("Phase 1: Self-Play")
        new_examples = []
        game_lengths = []

        for g in tqdm(range(cfg['games_per_iter']), desc="Games"):
            game_data = execute_episode(
                mcts, mapper, cfg['mcts_sims'],
                cfg['temp_threshold'], cfg['temp_final']
            )
            new_examples.extend(game_data)
            game_lengths.append(len(game_data))
            mcts.clear_cache()

        replay_buffer.extend(new_examples)
        avg_len = np.mean(game_lengths)
        print(f"  Buffer: {len(replay_buffer):,} | New: {len(new_examples):,} | Avg game: {avg_len:.1f}")

        # Phase 2: Training
        print("\nPhase 2: Training")
        avg_loss = train(
            model, optimizer, replay_buffer, device,
            cfg['epochs'], cfg['batch_size'], cfg['max_grad_norm']
        )

        # Phase 3: Arena Evaluation (compare against supervised baseline)
        if (iteration + 1) % cfg['arena_interval'] == 0:
            print("\nPhase 3: Arena Evaluation (vs supervised baseline)")
            model.eval()
            baseline_model.eval()

            win_rate = arena_compare(
                model, baseline_model, device, mapper,
                cfg['arena_games'], cfg['mcts_batch_size']
            )
            print(f"  New model win rate vs baseline: {win_rate:.1%}")

            if win_rate >= cfg['win_threshold']:
                print(f"  ✓ New model beats supervised baseline - accepted")
                best_model.load_state_dict(model.state_dict())
                torch.save(best_model.state_dict(), best_model_path)
            else:
                print(f"  ✗ New model doesn't beat baseline, reverting to best RL model")
                model.load_state_dict(best_model.state_dict())

        # Save checkpoint and latest model
        save_checkpoint(checkpoint_path, iteration + 1, model, optimizer, replay_buffer)
        torch.save(model.state_dict(), model_output_path)

        clear_device_cache(device)
        print(f"Iteration {iteration+1} complete | Loss: {avg_loss:.4f}")

    # Final save
    torch.save(model.state_dict(), model_output_path.replace('.pth', '_final.pth'))
    print("\nTraining complete!")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AlphaZero Chess Training")
    parser.add_argument("--supervised-model", default="supervised_chess_model.pth",
                        help="Path to supervised model (used as baseline)")
    parser.add_argument("--model-output", default="rl_chess_model_latest.pth")
    parser.add_argument("--best-model", default="rl_chess_model_best.pth")
    parser.add_argument("--checkpoint", default="rl_checkpoint.pth")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=500)
    parser.add_argument("--mcts-sims", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)

    args = parser.parse_args()

    config = {
        "iterations": args.iterations,
        "games_per_iter": args.games_per_iter,
        "mcts_sims": args.mcts_sims,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }

    run_training(
        supervised_model_path=args.supervised_model,
        model_output_path=args.model_output,
        best_model_path=args.best_model,
        checkpoint_path=args.checkpoint,
        config=config,
    )
