import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import chess
from collections import deque
import random
import gc
import os
from mcts import BatchedMCTS
from utils import ActionMapper, board_to_tensor, INPUT_PLANES
from nn import ChessNet

# --- Hyperparameters ---
ITERATIONS = 100           # Increased from 10 for more training
GAMES_PER_ITER = 100       # Increased from 50 for more data per iteration
MCTS_SIMS = 400            # Increased from 200 for better self-play quality
BATCH_SIZE = 256           # Batch size for training
EPOCHS = 25                # Training epochs per iteration
MAX_BUFFER_SIZE = 500000   # Increased replay buffer for more diverse data
MCTS_BATCH_SIZE = 32       # Batch size for MCTS neural net evaluation
CACHE_SIZE = 100000        # Transposition table size

# Learning rate schedule
INITIAL_LR = 0.2         
LR_DROPS = {
    50: 0.02,
    75: 0.002,
    90: 0.0002,
}

# Model architecture
NUM_RES_BLOCKS = 10 
NUM_CHANNELS = 128
USE_SE = True

# Arena settings for model comparison
ARENA_GAMES = 40
WIN_THRESHOLD = 0.55

# Temperature settings for move selection
TEMP_THRESHOLD = 30
TEMP_FINAL = 0.1

# Gradient clipping
MAX_GRAD_NORM = 1.0


def get_device():
    """Detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def execute_episode(mcts, mapper, temperature_threshold=TEMP_THRESHOLD):
    """
    Execute one full game of self-play to generate training data.

    Uses temperature-based move selection: high temperature early for exploration,
    low temperature later for exploitation.

    Args:
        mcts: BatchedMCTS instance used for move selection.
        mapper: ActionMapper for converting moves to policy vector indices.
        temperature_threshold: Number of moves to use temperature=1.

    Returns:
        List of training examples as (board_tensor, policy_vector, value) tuples.
    """
    examples = []
    board = chess.Board()
    mcts_root = None

    while not board.is_game_over():
        # 1. Run MCTS
        root = mcts.run_self_play_simulation(
            board, num_simulations=MCTS_SIMS, root=mcts_root
        )

        # 2. Extract visit counts as policy target
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        moves = list(root.children.keys())
        sum_visits = visit_counts.sum()

        if sum_visits == 0:
            break  # No valid moves

        # 3. Apply temperature to visit counts
        move_number = len(board.move_stack)
        if move_number < temperature_threshold:
            # Temperature = 1: sample proportionally to visit counts
            pi = visit_counts / sum_visits
            chosen_idx = np.random.choice(len(moves), p=pi)
        else:
            # Low temperature: nearly greedy but with slight randomness
            # Raise visit counts to power of 1/temperature
            temp = TEMP_FINAL
            adjusted = visit_counts ** (1.0 / temp)
            pi = adjusted / adjusted.sum()
            chosen_idx = np.random.choice(len(moves), p=pi)

        chosen_move = moves[chosen_idx]

        # Store training example (before making move)
        examples.append([board.copy(), moves, visit_counts / sum_visits, None])

        board.push(chosen_move)

        # Subtree reuse
        if chosen_move in root.children:
            mcts_root = root.children[chosen_move]
            mcts_root.parent = None
        else:
            mcts_root = None

    # 4. Game over - assign results
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        result = 0
    else:
        result = 1 if outcome.winner == chess.WHITE else -1

    processed_examples = []
    for state, moves, pi, _ in examples:
        # Result from perspective of player to move
        player_result = result if state.turn == chess.WHITE else -result

        # Convert policy to vector
        pi_vector = np.zeros(mapper.vocab_size, dtype=np.float32)
        for i, move in enumerate(moves):
            idx = mapper.encode(move)
            if idx is not None:
                pi_vector[idx] = pi[i]

        processed_examples.append((board_to_tensor(state), pi_vector, player_result))

    return processed_examples


def train(model, optimizer, scheduler, replay_buffer, device, epochs=EPOCHS):
    """
    Train the neural network on data from the replay buffer.

    Uses the optimizer and scheduler passed in (persisted across iterations).

    Args:
        model: The ChessNet model.
        optimizer: Persistent optimizer.
        scheduler: Learning rate scheduler.
        replay_buffer: Deque containing training examples.
        device: Compute device.
        epochs: Training epochs per iteration.

    Returns:
        Average loss over training.
    """
    mse = nn.MSELoss()
    model.train()

    if len(replay_buffer) < BATCH_SIZE:
        print(f"  Buffer too small ({len(replay_buffer)} < {BATCH_SIZE}), skipping training")
        return 0.0

    buffer_list = list(replay_buffer)
    batch_count = min(len(buffer_list) // BATCH_SIZE, 1000)

    print(f"  Training for {epochs} epochs on {len(buffer_list)} examples "
          f"({batch_count} batches/epoch, lr={optimizer.param_groups[0]['lr']:.6f})")

    total_epoch_loss = 0

    for epoch in range(epochs):
        total_loss = 0
        total_loss_v = 0
        total_loss_p = 0

        for _ in range(batch_count):
            batch = random.sample(buffer_list, BATCH_SIZE)

            state_tensor = torch.stack([b[0] for b in batch]).to(device)
            policy_target = torch.tensor(
                np.array([b[1] for b in batch]), dtype=torch.float32
            ).to(device)
            value_target = torch.tensor(
                np.array([b[2] for b in batch]), dtype=torch.float32
            ).to(device)

            # Forward pass
            p_pred, v_pred = model(state_tensor)

            # Loss calculation
            loss_v = mse(v_pred.squeeze(), value_target)
            loss_p = -torch.sum(policy_target * torch.log_softmax(p_pred, dim=1)) / BATCH_SIZE
            loss = loss_v + loss_p

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
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

    # Step the scheduler after all epochs
    if scheduler is not None:
        scheduler.step()

    return total_epoch_loss / epochs


def play_arena_game(model1, model2, device, mapper):
    """
    Play a single game between two models.

    Args:
        model1: First model (plays as white in odd games, black in even).
        model2: Second model.
        device: Compute device.
        mapper: ActionMapper.

    Returns:
        1 if model1 wins, -1 if model2 wins, 0 for draw.
    """
    mcts1 = BatchedMCTS(model1, device, mapper, batch_size=MCTS_BATCH_SIZE)
    mcts2 = BatchedMCTS(model2, device, mapper, batch_size=MCTS_BATCH_SIZE)

    board = chess.Board()

    while not board.is_game_over():
        # Determine which model plays
        if board.turn == chess.WHITE:
            mcts = mcts1
        else:
            mcts = mcts2

        # Search and pick best move
        root = mcts.search(board, num_simulations=100)
        if not root.children:
            break

        best_move = max(root.children.keys(), key=lambda m: root.children[m].visit_count)
        board.push(best_move)

        # Clear caches to avoid memory bloat
        mcts1.clear_cache()
        mcts2.clear_cache()

    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 0
    return 1 if outcome.winner == chess.WHITE else -1


def arena_compare(new_model, old_model, device, mapper, num_games=ARENA_GAMES):
    """
    Compare two models by playing games between them.

    Each model plays half the games as white and half as black.

    Args:
        new_model: Candidate model.
        old_model: Current best model.
        device: Compute device.
        mapper: ActionMapper.
        num_games: Total games to play.

    Returns:
        Win rate of new_model (0.0 to 1.0).
    """
    new_wins = 0
    old_wins = 0
    draws = 0

    for i in range(num_games):
        # Alternate colors
        if i % 2 == 0:
            result = play_arena_game(new_model, old_model, device, mapper)
            if result == 1:
                new_wins += 1
            elif result == -1:
                old_wins += 1
            else:
                draws += 1
        else:
            result = play_arena_game(old_model, new_model, device, mapper)
            if result == 1:
                old_wins += 1
            elif result == -1:
                new_wins += 1
            else:
                draws += 1

        print(f"    Arena game {i+1}/{num_games}: "
              f"New={new_wins}, Old={old_wins}, Draws={draws}")

    total_played = new_wins + old_wins + draws
    # Win rate: wins + 0.5*draws
    win_rate = (new_wins + 0.5 * draws) / total_played if total_played > 0 else 0.5
    return win_rate


def update_learning_rate(optimizer, iteration):
    """Update learning rate based on iteration count."""
    for threshold, lr in sorted(LR_DROPS.items()):
        if iteration >= threshold:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def main():
    """
    Main entry point for AlphaZero-style reinforcement learning training.

    Improvements over basic implementation:
    - Persistent optimizer with momentum across iterations
    - Learning rate schedule with drops at key points
    - Arena evaluation to only keep improving models
    - Gradient clipping for stability
    - Temperature-based move selection
    - Larger model architecture
    """
    device = get_device()
    print(f"AlphaZero Training on {device}")
    print(f"Config: {GAMES_PER_ITER} games/iter, {MCTS_SIMS} sims, "
          f"batch={MCTS_BATCH_SIZE}, res_blocks={NUM_RES_BLOCKS}, channels={NUM_CHANNELS}")

    mapper = ActionMapper()

    # Initialize model with larger architecture
    model = ChessNet(
        action_size=mapper.vocab_size,
        input_channels=INPUT_PLANES,
        num_res_blocks=NUM_RES_BLOCKS,
        num_channels=NUM_CHANNELS,
        use_se=USE_SE
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Load pretrained weights if available
    if os.path.exists("supervised_chess_model.pth"):
        try:
            # Try loading, but architecture might not match
            state_dict = torch.load("supervised_chess_model.pth", map_location=device)
            model.load_state_dict(state_dict)
            print("Loaded pretrained supervised model")
        except Exception as e:
            print(f"Could not load pretrained model (architecture mismatch?): {e}")
            print("Starting from scratch")
    elif os.path.exists("rl_chess_model_best.pth"):
        model.load_state_dict(torch.load("rl_chess_model_best.pth", map_location=device))
        print("Resuming from best RL model")

    # Keep a copy of the best model for arena comparison
    best_model = ChessNet(
        action_size=mapper.vocab_size,
        input_channels=INPUT_PLANES,
        num_res_blocks=NUM_RES_BLOCKS,
        num_channels=NUM_CHANNELS,
        use_se=USE_SE
    ).to(device)
    best_model.load_state_dict(model.state_dict())

    # Persistent optimizer (SGD with momentum like AlphaZero)
    optimizer = optim.SGD(
        model.parameters(),
        lr=INITIAL_LR,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )

    # Learning rate scheduler (can also use manual drops via update_learning_rate)
    scheduler = None  # We'll use manual LR drops instead

    replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)
    mcts = BatchedMCTS(model, device, mapper, batch_size=MCTS_BATCH_SIZE, cache_size=CACHE_SIZE)

    for iteration in range(ITERATIONS):
        print(f"\n{'='*60}")
        print(f"--- Iteration {iteration+1}/{ITERATIONS} ---")
        print(f"{'='*60}")

        # Update learning rate based on schedule
        update_learning_rate(optimizer, iteration)

        # Phase 1: Self-Play
        print("Phase 1: Self-Play")
        new_examples = []
        game_lengths = []

        for g in range(GAMES_PER_ITER):
            game_data = execute_episode(mcts, mapper)
            new_examples.extend(game_data)
            game_lengths.append(len(game_data))
            mcts.clear_cache()

            if (g + 1) % 10 == 0:
                avg_len = np.mean(game_lengths[-10:])
                print(f"  Games: {g+1}/{GAMES_PER_ITER} | Avg length: {avg_len:.1f}")

        replay_buffer.extend(new_examples)
        print(f"  Buffer: {len(replay_buffer):,} | New examples: {len(new_examples):,}")

        # Phase 2: Training
        print("\nPhase 2: Training")
        avg_loss = train(model, optimizer, scheduler, replay_buffer, device, EPOCHS)

        # Phase 3: Arena Evaluation (every 5 iterations to save time)
        if (iteration + 1) % 5 == 0 and iteration > 0:
            print("\nPhase 3: Arena Evaluation")
            model.eval()
            best_model.eval()

            win_rate = arena_compare(model, best_model, device, mapper, num_games=ARENA_GAMES)
            print(f"  New model win rate: {win_rate:.1%}")

            if win_rate >= WIN_THRESHOLD:
                print(f"  New model accepted (>= {WIN_THRESHOLD:.0%})")
                best_model.load_state_dict(model.state_dict())
                torch.save(best_model.state_dict(), "rl_chess_model_best.pth")
            else:
                print(f"  New model rejected (< {WIN_THRESHOLD:.0%}), reverting")
                model.load_state_dict(best_model.state_dict())
        else:
            # Save checkpoint without arena
            torch.save(model.state_dict(), "rl_chess_model_latest.pth")

        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()

        print(f"Iteration {iteration+1} complete | Loss: {avg_loss:.4f}")

    # Final save
    torch.save(model.state_dict(), "rl_chess_model_final.pth")
    print("\nTraining complete! Models saved.")


if __name__ == "__main__":
    main()
