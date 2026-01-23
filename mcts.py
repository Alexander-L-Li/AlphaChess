import math
import torch
import numpy as np
from utils import board_to_tensor

class Node:
    def __init__(self, state, parent=None, action_taken=None):
        self.state = state  # The chess.Board object
        self.parent = parent
        self.action_taken = action_taken # Move used to get here
        self.children = {} # Map move -> Node
        
        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0 # P(s, a) from neural net
        
    def is_expanded(self):
        return len(self.children) > 0
        
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def select_child(self, cpuct=1.0):
        # UCB Selection
        best_score = -float('inf')
        best_child = None
        
        for action, child in self.children.items():
            q_value = -child.value() # Negate because child's win is our loss
            
            # UCB formula
            u_value = cpuct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def expand(self, policy_preds):
        # policy_preds: list of (move, prob) from Neural Net
        for move, prob in policy_preds:
            if move not in self.children:
                # Create new node
                new_state = self.state.copy()
                new_state.push(move)
                child = Node(new_state, parent=self, action_taken=move)
                child.prior = prob
                self.children[move] = child

    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            # Switch perspective: If valid for me is +1, it is -1 for my parent
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, model, device, mapper):
        self.model = model
        self.device = device
        self.mapper = mapper
        
    def search(self, root_state, num_simulations=100):
        root = Node(root_state)
        
        for _ in range(num_simulations):
            node = root
            
            # 1. Selection
            while node.is_expanded():
                node = node.select_child()
                
            # Check if game ended
            if node.state.is_game_over():
                # Get true result: 1=win, -1=loss, 0=draw
                # Note: This logic needs careful implementation based on whose turn it is
                value = self.get_game_result(node.state)
                node.backpropagate(value)
                continue
            
            # 2. Evaluation (Neural Net)
            policy_preds, value = predict_masked(self.model, node.state, self.device, self.mapper)
                
            # 3. Expansion
            node.expand(policy_preds)
            
            # 4. Backpropagation
            node.backpropagate(value)
            
        return root # Return the tree to pick the best move
        
    def get_game_result(self, board):
        # Simplified result handling
        outcome = board.outcome()
        if outcome.winner is None: return 0
        if outcome.winner == board.turn: return 1
        return -1

    def run_self_play_simulation(self, root_state, num_simulations=100):
        """
        Special search for training: Adds noise to the root to encourage exploration.
        """
        root = Node(root_state)
        
        # 1. Add Dirichlet Noise to Root Node (Only at the start of search)
        # This makes the model try moves it might otherwise ignore
        self._add_dirichlet_noise(root, root_state)

        # 2. Standard Search Loop
        for _ in range(num_simulations):
            node = root
            while node.is_expanded():
                node = node.select_child()
            
            if node.state.is_game_over():
                value = self.get_game_result(node.state)
                node.backpropagate(value)
                continue
            
            policy_preds, value = predict_masked(self.model, node.state, self.device, self.mapper)
            node.expand(policy_preds)
            node.backpropagate(value)
            
        return root

    def _add_dirichlet_noise(self, node, board):
        """
        AlphaZero mechanic: Adds random noise to the prior probabilities P(s,a)
        at the root node.
        """
        # Ensure node is expanded first so we have children
        policy_preds, _ = predict_masked(self.model, board, self.device, self.mapper)
        node.expand(policy_preds)
        
        # Parameters used by AlphaZero for Chess
        epsilon = 0.25  # How much noise to mix in (25% noise, 75% original net)
        alpha = 0.3     # Shape of noise (0.3 is standard for Chess)
        
        moves = list(node.children.keys())
        noise = np.random.dirichlet([alpha] * len(moves))
        
        for i, move in enumerate(moves):
            child = node.children[move]
            # Mix: New_Prior = (0.75 * Old_Prior) + (0.25 * Noise)
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


def predict_masked(model, board, device, mapper):
    """
    Runs the model and returns a dictionary {chess.Move: probability}
    only for currently LEGAL moves.
    """
    model.eval()
    
    # 1. Prepare Input
    tensor = board_to_tensor(board).unsqueeze(0).to(device)
    
    # 2. Forward Pass
    with torch.no_grad():
        policy_logits, value = model(tensor)
    
    policy_logits = policy_logits.squeeze(0).cpu().numpy() # Shape: (vocab_size,)
    
    # 3. Create Mask for Legal Moves
    legal_moves = list(board.legal_moves)
    legal_indices = []
    
    for move in legal_moves:
        idx = mapper.encode(move)
        if idx is not None:
            legal_indices.append(idx)
    
    if not legal_indices:
        return [], value.item()

    # 4. Filter and Softmax
    # We only care about the logits for legal moves
    legal_logits = policy_logits[legal_indices]
    
    # Numerical stability shift (subtract max to prevent overflow)
    exp_logits = np.exp(legal_logits - np.max(legal_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # 5. Map back to chess.Move objects
    move_probs = []
    for i, move in enumerate(legal_moves):
        if 0 <= i < len(probs):
            move_probs.append((move, probs[i]))
        
    return move_probs, value.item()

    