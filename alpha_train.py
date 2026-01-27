import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import chess
from collections import deque
import random
import gc
from mcts import BatchedMCTS
from utils import ActionMapper, board_to_tensor
from nn import ChessNet

# --- Hyperparameters (Optimized for 36GB MacBook) ---
ITERATIONS = 10        # Total loops of Self-Play -> Train
GAMES_PER_ITER = 50   # Games to play per loop
MCTS_SIMS = 200        # Reduced sims (early stopping compensates)
BATCH_SIZE = 256       # Larger batch for GPU efficiency
EPOCHS = 10            # Training epochs on the new data per iteration
MAX_BUFFER_SIZE = 100000  # Large replay buffer (~2GB RAM)
MCTS_BATCH_SIZE = 32   # Batch size for MCTS neural net evaluation
CACHE_SIZE = 50000     # Transposition table size


def get_device():
    """Detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def execute_episode(mcts, mapper):
    """
    Execute one full game of self-play to generate training data.

    Plays a complete game from the starting position using MCTS to select moves.
    For the first 30 moves, moves are sampled proportionally to visit counts
    to encourage exploration. After that, the most-visited move is chosen.
    Implements subtree reuse to speed up consecutive MCTS searches.

    Args:
        mcts: BatchedMCTS instance used for move selection.
        mapper: ActionMapper for converting moves to policy vector indices.

    Returns:
        List of training examples as (board_tensor, policy_vector, value) tuples,
        where value is +1 for white win, -1 for black win, 0 for draw,
        adjusted to the perspective of the player to move in each position.
    """
    examples = []
    board = chess.Board()
    mcts_root = None

    while not board.is_game_over():
        # 1. Run MCTS (optimize by using previous root)
        root = mcts.run_self_play_simulation(board, num_simulations=MCTS_SIMS, root=mcts_root)
        
        # 2. Extract data
        visit_counts = [child.visit_count for child in root.children.values()]
        moves = [move for move in root.children.keys()]
        sum_visits = sum(visit_counts)
        pi = [v / sum_visits for v in visit_counts]
        
        examples.append([board.copy(), moves, pi, None])
        
        # 3. Pick move
        if len(board.move_stack) < 30:
            chosen_move = np.random.choice(moves, p=pi)
        else:
            chosen_move = moves[np.argmax(visit_counts)]
            
        board.push(chosen_move)
        
        # Subtree reuse optimization
        if chosen_move in root.children:
            mcts_root = root.children[chosen_move]
            mcts_root.parent = None  # Detach from old tree to allow garbage collection
        else:
            mcts_root = None  # Fallback but shouldn't happen in standard self-play
            
    # 4. Game over logic
    outcome = board.outcome()
    if outcome.winner is None: result = 0
    else: result = 1 if outcome.winner == chess.WHITE else -1
        
    processed_examples = []
    for state, moves, pi, _ in examples:
        if state.turn == chess.WHITE: player_result = result
        else: player_result = -result
            
        pi_vector = np.zeros(mapper.vocab_size, dtype=np.float32)
        for i, move in enumerate(moves):
            idx = mapper.encode(move)
            if idx is not None: pi_vector[idx] = pi[i]
                
        processed_examples.append((board_to_tensor(state), pi_vector, player_result))
        
    return processed_examples

def train(model, replay_buffer, device, epochs=10):
    """
    Trains the neural network on data from the replay buffer.

    Args:
        model: The ChessNet model.
        replay_buffer: Deque containing training examples.
        device: CPU or MPS device.
        epochs: How many times to pass over the data (Hyperparameter).
    """
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    mse = nn.MSELoss()

    model.train()

    if len(replay_buffer) < BATCH_SIZE:
        print(f"  Buffer too small ({len(replay_buffer)} < {BATCH_SIZE}), skipping training")
        return

    # Convert to list for faster random sampling with large buffers
    buffer_list = list(replay_buffer)

    # Batches per epoch (cap at reasonable number for very large buffers)
    batch_count = min(len(buffer_list) // BATCH_SIZE, 500)

    print(f"  Training for {epochs} epochs on {len(buffer_list)} examples ({batch_count} batches/epoch)...")

    for epoch in range(epochs):
        total_loss = 0
        total_loss_v = 0
        total_loss_p = 0

        for _ in range(batch_count):
            # Sample a random batch
            batch = random.sample(buffer_list, BATCH_SIZE)

            # Unpack data
            state_tensor = torch.stack([b[0] for b in batch]).to(device)
            policy_target = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32).to(device)
            value_target = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).to(device)

            # Forward pass
            p_pred, v_pred = model(state_tensor)

            # Calculate loss
            loss_v = mse(v_pred.squeeze(), value_target)
            loss_p = -torch.sum(policy_target * torch.log_softmax(p_pred, dim=1)) / BATCH_SIZE
            loss = loss_v + loss_p

            # Backward pass
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

def main():
    """
    Main entry point for AlphaZero-style reinforcement learning training.

    Runs iterative self-play and neural network training cycles to improve
    the chess model. Saves the trained model after each iteration.
    """
    device = get_device()
    print(f"RL Training on {device}")
    print(f"Config: {GAMES_PER_ITER} games/iter, {MCTS_SIMS} sims, batch={MCTS_BATCH_SIZE}")

    mapper = ActionMapper()
    model = ChessNet(action_size=mapper.vocab_size).to(device)
    model.load_state_dict(torch.load("supervised_chess_model.pth"))

    replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)
    mcts = BatchedMCTS(model, device, mapper, batch_size=MCTS_BATCH_SIZE, cache_size=CACHE_SIZE)

    for i in range(ITERATIONS):
        print(f"\n{'='*50}")
        print(f"--- Iteration {i+1}/{ITERATIONS} ---")
        print(f"{'='*50}")

        # Phase 1: Self-Play
        print("Self-Playing...")
        new_examples = []
        for g in range(GAMES_PER_ITER):
            game_data = execute_episode(mcts, mapper)
            new_examples.extend(game_data)
            mcts.clear_cache()  # Clear transposition table between games
            print(f"  Games completed: {g+1}/{GAMES_PER_ITER}")

        replay_buffer.extend(new_examples)
        print(f"Buffer size: {len(replay_buffer)} | New examples: {len(new_examples)}")

        # Phase 2: Training (Update Neural Net)
        print("Training Network...")
        train(model, replay_buffer, device, EPOCHS)

        # Device memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        gc.collect()

        # Save current best
        torch.save(model.state_dict(), "rl_chess_model_latest.pth")
        print(f"Model saved to rl_chess_model_latest.pth")

if __name__ == "__main__":
    main()