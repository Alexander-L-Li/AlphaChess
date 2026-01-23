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
ITERATIONS = 10        # Total loops of Self-Play -> Train
GAMES_PER_ITER = 50    # Games to play per loop (Increase if you have time)
MCTS_SIMS = 100         # Lower sims for speed (Standard is 800, too slow for Mac)
BATCH_SIZE = 64
EPOCHS = 10             # Training epochs on the new data per iteration
MAX_BUFFER_SIZE = 2500 # Keep last 5000 games in memory (Replay Buffer)

def execute_episode(mcts, mapper):
    """
    Plays one single game of Self-Play (Model vs Model).
    Returns: A list of training examples [(board_tensor, policy_target, value_target)]
    """
    examples = []
    board = chess.Board()
    mcts_root = None # Can be reused in optimized versions, but reset for now
    
    while not board.is_game_over():
        # 1. Run MCTS with Noise
        root = mcts.run_self_play_simulation(board, num_simulations=MCTS_SIMS)
        
        # 2. Extract Training Data (The "Improved" Policy)
        # We want the Neural Net to predict the Visit Counts (N), not the raw Prior (P)
        visit_counts = [child.visit_count for child in root.children.values()]
        moves = [move for move in root.children.keys()]
        
        # Normalize visits to get a probability distribution (pi)
        sum_visits = sum(visit_counts)
        pi = [v / sum_visits for v in visit_counts]
        
        # Store data: (Board State, Target Policy, Placeholder Value)
        # We don't know the Winner yet, so we put None for value
        examples.append([board.copy(), moves, pi, None])
        
        # 3. Pick a move
        # Early in the game (first 30 moves), pick randomly based on 'pi' (Temperature=1)
        # Later in the game, pick the best move deterministically (Temperature=0)
        if len(board.move_stack) < 30:
            chosen_move = np.random.choice(moves, p=pi)
        else:
            chosen_move = moves[np.argmax(visit_counts)]
            
        board.push(chosen_move)
        
    # 4. Game Ended - Assign Value
    outcome = board.outcome()
    if outcome.winner is None: 
        result = 0 # Draw
    else:
        # 1 if White won, -1 if Black won
        result = 1 if outcome.winner == chess.WHITE else -1
        
    # 5. Backfill the result to all examples
    processed_examples = []
    for state, moves, pi, _ in examples:
        # Perspective: If 'state' turn was White, and White won, value is +1.
        # If 'state' turn was Black, and White won, value is -1 (Loss for Black).
        if state.turn == chess.WHITE:
            player_result = result
        else:
            player_result = -result
            
        # Convert moves/pi to the full vocabulary vector
        pi_vector = np.zeros(mapper.vocab_size, dtype=np.float32)
        for i, move in enumerate(moves):
            idx = mapper.encode(move)
            if idx is not None:
                pi_vector[idx] = pi[i]
                
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
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
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
        
        # Phase 1: Self-Play (Gather Data)
        print("Self-Playing...", end="", flush=True)
        new_examples = []
        for g in range(GAMES_PER_ITER):
            game_data = execute_episode(mcts, mapper)
            new_examples.extend(game_data)
            print(f".", end="", flush=True)

        replay_buffer.extend(new_examples)
        print(f"\nBuffer size: {len(replay_buffer)}")
        
        # Phase 2: Training (Update Neural Net)
        print("Training Network...")
        train(model, replay_buffer, device, EPOCHS)
        
        # Save current best
        torch.save(model.state_dict(), "rl_chess_model_latest.pth")

if __name__ == "__main__":
    main()