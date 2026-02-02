import math
from collections import OrderedDict
import torch
import numpy as np
from utils import board_to_tensor

class Node:
    """Represents a node in the MCTS search tree."""

    def __init__(self, state, parent=None, action_taken=None):
        """
        Initialize a new MCTS node.

        Args:
            state: The chess.Board object representing this position.
            parent: Parent Node in the tree (None for root).
            action_taken: The chess.Move used to reach this node from parent.
        """
        self.state = state  # The chess.Board object
        self.parent = parent
        self.action_taken = action_taken # Move used to get here
        self.children = {} # Map move -> Node

        self.visit_count = 0
        self.value_sum = 0
        self.prior = 0 # P(s, a) from neural net
        self.virtual_loss = 0  # For batched MCTS

    def is_expanded(self):
        """Check if this node has been expanded (has children)."""
        return len(self.children) > 0

    def value(self):
        """
        Calculate the average value of this node.

        Returns:
            The mean value, adjusted for virtual loss during batched MCTS.
        """
        if self.visit_count == 0:
            return 0
        # Include virtual loss in value calculation (makes node look worse during batching)
        total_visits = self.visit_count + self.virtual_loss
        adjusted_value = self.value_sum - self.virtual_loss  # Virtual loss counts as losses
        return adjusted_value / total_visits

    def add_virtual_loss(self, amount=1):
        """Apply virtual loss to discourage parallel selection of same path."""
        self.virtual_loss += amount

    def remove_virtual_loss(self, amount=1):
        """Remove virtual loss after backpropagation."""
        self.virtual_loss -= amount

    def select_child(self, cpuct=1.0):
        """
        Select the best child node using the UCB formula.

        Args:
            cpuct: Exploration constant for UCB calculation.

        Returns:
            The child Node with the highest UCB score.
        """
        # UCB Selection (accounts for virtual loss)
        best_score = -float('inf')
        best_child = None

        # Use total visits including virtual loss for parent
        parent_visits = self.visit_count + self.virtual_loss

        for action, child in self.children.items():
            q_value = -child.value()  # Negate because child's win is our loss

            # UCB formula with virtual loss accounted for
            child_visits = child.visit_count + child.virtual_loss
            u_value = cpuct * child.prior * (math.sqrt(parent_visits) / (1 + child_visits))

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand(self, policy_preds):
        """
        Expand this node by creating child nodes for all legal moves.

        Args:
            policy_preds: List of (move, probability) tuples from the neural network.
        """
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
        """
        Backpropagate a value up the tree from this node to the root.

        Args:
            value: The value to propagate (flipped at each level for alternating players).
        """
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            # Switch perspective: If valid for me is +1, it is -1 for my parent
            self.parent.backpropagate(-value)


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
    
    # Numerical stability shift (subtract max to prevent overflow) s
    exp_logits = np.exp(legal_logits - np.max(legal_logits))
    probs = exp_logits / np.sum(exp_logits)
    
    # 5. Map back to chess.Move objects
    move_probs = []
    for i, move in enumerate(legal_moves):
        if 0 <= i < len(probs):
            move_probs.append((move, probs[i]))

    return move_probs, value.item()


class EvaluationCache:
    """LRU cache for neural network position evaluations (transposition table)."""

    def __init__(self, max_size=50000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def _get_key(self, board):
        """Generate cache key from board position (FEN without move counters)."""
        fen_parts = board.fen().split()
        # Keep piece placement, turn, castling, en passant (ignore halfmove/fullmove)
        return ' '.join(fen_parts[:4])

    def get(self, board):
        """Retrieve cached evaluation if available. O(1) operation."""
        key = self._get_key(board)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, board, policy_probs, value):
        """Store evaluation in cache. O(1) operation."""
        key = self._get_key(board)

        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (policy_probs, value)
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        """Clear the cache (call between games to prevent memory bloat)."""
        self.cache.clear()


class BatchedMCTS:
    """
    Optimized MCTS with batched neural network evaluation.

    Key optimizations:
    - Collects multiple leaf nodes before evaluating them in a single batch
    - Uses virtual loss to prevent duplicate path selection during batching
    - Includes transposition table to avoid re-evaluating identical positions
    - Supports early stopping when a dominant move emerges
    """

    def __init__(self, model, device, mapper, batch_size=32, cache_size=50000):
        self.model = model
        self.device = device
        self.mapper = mapper
        self.batch_size = batch_size
        self.eval_cache = EvaluationCache(max_size=cache_size)

    def run_self_play_simulation(self, root_state, num_simulations=400, root=None,
                                  early_stop_threshold=0.95, min_simulations=100):
        """
        Run MCTS with batched evaluation and early stopping.

        Args:
            root_state: The chess.Board position to search from
            num_simulations: Maximum number of simulations
            root: Optional existing root node for subtree reuse
            early_stop_threshold: Stop early if one move has this fraction of visits
            min_simulations: Minimum simulations before early stopping allowed
        """
        if root is None:
            root = Node(root_state)

        # Expand root with Dirichlet noise (uses cached value)
        self._add_dirichlet_noise(root, root_state)

        simulations_done = 0
        while simulations_done < num_simulations:
            # Collect a batch of leaf nodes
            leaves = []
            paths = []
            terminal_values = []

            batch_count = min(self.batch_size, num_simulations - simulations_done)

            for _ in range(batch_count):
                node = root
                path = [node]

                # Selection with virtual loss
                while node.is_expanded():
                    node.add_virtual_loss()
                    node = node.select_child()
                    path.append(node)

                node.add_virtual_loss()

                if node.state.is_game_over():
                    # Handle terminal states immediately
                    value = self._get_game_result(node.state)
                    terminal_values.append((path, value))
                else:
                    leaves.append(node)
                    paths.append(path)

            # Backpropagate terminal states
            for path, value in terminal_values:
                self._backpropagate_with_virtual_loss(path, value)

            # Batch evaluation of non-terminal leaves
            if leaves:
                policies, values = self._batch_predict(leaves)

                for i, (leaf, path) in enumerate(zip(leaves, paths)):
                    leaf.expand(policies[i])
                    self._backpropagate_with_virtual_loss(path, values[i])

            simulations_done += batch_count

            # Early stopping check
            if simulations_done >= min_simulations:
                if self._should_early_stop(root, early_stop_threshold):
                    break

        return root

    def search(self, root_state, num_simulations=200):
        """Standard search without Dirichlet noise (for play mode)."""
        root = Node(root_state)

        # Initial expansion
        cached = self.eval_cache.get(root_state)
        if cached:
            policy_probs, value = cached
        else:
            policy_probs, value = predict_masked(self.model, root_state, self.device, self.mapper)
            self.eval_cache.put(root_state, policy_probs, value)
        root.expand(policy_probs)
        root.value_sum = value
        root.visit_count = 1

        simulations_done = 1
        while simulations_done < num_simulations:
            batch_count = min(self.batch_size, num_simulations - simulations_done)
            leaves = []
            paths = []
            terminal_values = []

            for _ in range(batch_count):
                node = root
                path = [node]

                while node.is_expanded():
                    node.add_virtual_loss()
                    node = node.select_child()
                    path.append(node)

                node.add_virtual_loss()

                if node.state.is_game_over():
                    value = self._get_game_result(node.state)
                    terminal_values.append((path, value))
                else:
                    leaves.append(node)
                    paths.append(path)

            for path, value in terminal_values:
                self._backpropagate_with_virtual_loss(path, value)

            if leaves:
                policies, values = self._batch_predict(leaves)
                for i, (leaf, path) in enumerate(zip(leaves, paths)):
                    leaf.expand(policies[i])
                    self._backpropagate_with_virtual_loss(path, values[i])

            simulations_done += batch_count

        return root

    def _batch_predict(self, nodes):
        """Evaluate multiple nodes in a single forward pass with caching."""
        cached_results = []
        uncached_nodes = []
        uncached_indices = []

        # Check cache first
        for i, node in enumerate(nodes):
            cached = self.eval_cache.get(node.state)
            if cached is not None:
                cached_results.append((i, cached[0], cached[1]))
            else:
                uncached_nodes.append(node)
                uncached_indices.append(i)

        # Batch evaluate uncached positions
        uncached_policies = []
        uncached_values = []

        if uncached_nodes:
            self.model.eval()

            # Stack all board tensors
            tensors = torch.stack([
                board_to_tensor(node.state) for node in uncached_nodes
            ]).to(self.device)

            with torch.no_grad():
                policy_logits_batch, values_batch = self.model(tensors)

            # Process each result
            for j, node in enumerate(uncached_nodes):
                logits = policy_logits_batch[j]
                value = values_batch[j].item()

                # Get legal moves
                legal_moves = list(node.state.legal_moves)
                legal_indices = []
                valid_moves = []

                for move in legal_moves:
                    idx = self.mapper.encode(move)
                    if idx is not None:
                        legal_indices.append(idx)
                        valid_moves.append(move)

                if legal_indices:
                    legal_indices_tensor = torch.tensor(legal_indices, device=self.device)
                    legal_logits = logits[legal_indices_tensor]
                    probs = torch.softmax(legal_logits, dim=0).cpu().numpy()
                    move_probs = [(valid_moves[k], probs[k]) for k in range(len(valid_moves))]
                else:
                    move_probs = []

                uncached_policies.append(move_probs)
                uncached_values.append(value)

                # Cache the result
                self.eval_cache.put(node.state, move_probs, value)

        # Reconstruct results in original order
        policies = [None] * len(nodes)
        values = [None] * len(nodes)

        for i, policy, value in cached_results:
            policies[i] = policy
            values[i] = value

        for j, idx in enumerate(uncached_indices):
            policies[idx] = uncached_policies[j]
            values[idx] = uncached_values[j]

        return policies, values

    def _backpropagate_with_virtual_loss(self, path, value):
        """Backpropagate value and remove virtual loss from path."""
        current_value = value
        for node in reversed(path):
            node.remove_virtual_loss()
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value  # Flip perspective

    def _add_dirichlet_noise(self, node, board):
        """Expand root with Dirichlet noise for exploration."""
        # Check cache first
        cached = self.eval_cache.get(board)
        if cached:
            policy_probs, value = cached
        else:
            policy_probs, value = predict_masked(self.model, board, self.device, self.mapper)
            self.eval_cache.put(board, policy_probs, value)

        node.expand(policy_probs)
        node.value_sum = value  # Use the value instead of discarding
        node.visit_count = 1

        # Apply Dirichlet noise
        epsilon = 0.25
        alpha = 0.3

        moves = list(node.children.keys())
        if not moves:
            return

        noise = np.random.dirichlet([alpha] * len(moves))

        for i, move in enumerate(moves):
            child = node.children[move]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def _should_early_stop(self, root, threshold):
        """Check if a dominant move has emerged."""
        if not root.children:
            return False

        total_visits = sum(c.visit_count for c in root.children.values())
        if total_visits == 0:
            return False

        max_visits = max(c.visit_count for c in root.children.values())
        return (max_visits / total_visits) >= threshold

    def _get_game_result(self, board):
        """Get game result from the perspective of the player to move."""
        outcome = board.outcome()
        if outcome.winner is None:
            return 0
        if outcome.winner == board.turn:
            return 1
        return -1

    def clear_cache(self):
        """Clear the evaluation cache (call between games)."""
        self.eval_cache.clear()
