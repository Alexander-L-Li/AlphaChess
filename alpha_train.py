import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import chess
from collections import deque
import random
from mcts import MCTS
from utils import ActionMapper, board_to_tensor
from nn import ChessNet

# --- Hyperparameters ---
ITERATIONS = 5        # Total loops of Self-Play -> Train
GAMES_PER_ITER = 50    # Games to play per loop (Increase if you have time)
MCTS_SIMS = 800         # Lower sims for speed (Standard is 800, too slow for Mac)
BATCH_SIZE = 64
EPOCHS = 10            # Training epochs on the new data per iteration
MAX_BUFFER_SIZE = 250 # Keep last 5000 games in memory (Replay Buffer)

def execute_episode(mcts, mapper):
    examples = []
    board = chess.Board()
    mcts_root = None  # Start with no tree

    while not board.is_game_over():
        # 1. Run MCTS (Warm Start)
        root = mcts.run_self_play_simulation(board, num_simulations=MCTS_SIMS, root=mcts_root)
        
        # 2. Extract Data
        visit_counts = [child.visit_count for child in root.children.values()]
        moves = [move for move in root.children.keys()]
        sum_visits = sum(visit_counts)
        pi = [v / sum_visits for v in visit_counts]
        
        examples.append([board.copy(), moves, pi, None])
        
        # 3. Pick Move
        if len(board.move_stack) < 30:
            chosen_move = np.random.choice(moves, p=pi)
        else:
            chosen_move = moves[np.argmax(visit_counts)]
            
        board.push(chosen_move)
        
        # --- OPTIMIZATION: Subtree Reuse ---
        if chosen_move in root.children:
            mcts_root = root.children[chosen_move]
            mcts_root.parent = None  # Detach from old tree to allow garbage collection
        else:
            mcts_root = None  # Fallback (shouldn't happen in standard self-play)
            
    # 4. Game Over Logic (Unchanged)
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
        return
    
    # batches per epoch
    batch_count = len(replay_buffer) // BATCH_SIZE
    
    print(f"Training for {epochs} epochs on {len(replay_buffer)} examples...")

    for epoch in range(epochs):
        total_loss = 0
        
        for _ in range(batch_count):
            # Sample a random batch
            batch = random.sample(replay_buffer, BATCH_SIZE)
            
            # Unpack data
            state_tensor = torch.stack([b[0] for b in batch]).to(device)
            policy_target = torch.tensor(np.array([b[1] for b in batch])).to(device)
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
            
        avg_loss = total_loss / batch_count
        print(f"  Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"RL Training on {device}")
    
    mapper = ActionMapper()
    model = ChessNet(action_size=mapper.vocab_size).to(device)
    model.load_state_dict(torch.load("supervised_chess_model.pth"))
    
    replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)
    mcts = MCTS(model, device, mapper)
    
    for i in range(ITERATIONS):
        print(f"--- Iteration {i+1}/{ITERATIONS} ---")
        
        # Phase 1: Self-Play
        print("Self-Playing...")
        new_examples = []
        for g in range(GAMES_PER_ITER):
            game_data = execute_episode(mcts, mapper)
            new_examples.extend(game_data)
            print(f"Game {g} Complete")

        replay_buffer.extend(new_examples)
        print(f"\nBuffer size: {len(replay_buffer)}")
        
        # Phase 2: Training (Update Neural Net)
        print("Training Network...")
        train(model, replay_buffer, device, EPOCHS)
        
        # Save current best
        torch.save(model.state_dict(), "rl_chess_model_latest.pth")

if __name__ == "__main__":
    main()