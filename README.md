# AlphaChess

An AlphaZero-style chess engine built from scratch using PyTorch. The engine combines a convolutional neural network with Monte Carlo Tree Search (MCTS) to play chess.

## Features

- **Neural Network**: ResNet-style architecture with separate policy and value heads
- **MCTS**: Monte Carlo Tree Search with UCB selection and Dirichlet noise for exploration
- **Training Pipeline**:
  - Supervised learning on Lichess game data
  - Self-play reinforcement learning (AlphaZero approach)
- **Interactive GUI**: Pygame-based interface to play against the engine

## Project Structure

```
├── nn.py           # Neural network architecture (ChessNet)
├── mcts.py         # Monte Carlo Tree Search implementation
├── train.py        # Supervised training on Lichess data
├── alpha_train.py  # Self-play reinforcement learning
├── play.py         # Pygame GUI to play against the engine
├── utils.py        # Board encoding and action mapping utilities
├── dataset.py      # Dataset loader for training data
├── datagen.py      # Data generation utilities
└── assets/         # Chess piece images for GUI
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

1. **Supervised Pre-training** (optional but recommended):
   ```bash
   python train.py
   ```

2. **Self-Play Reinforcement Learning**:
   ```bash
   python alpha_train.py
   ```

### Playing Against the Engine

```bash
python play.py
```

Click to select a piece, then click the destination square to move. The AI will respond automatically.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- python-chess
- pygame
- numpy

## How It Works

1. **Board Representation**: The board is encoded as a 13-channel 8x8 tensor (12 piece types + turn indicator)

2. **Neural Network**: Takes the board state and outputs:
   - Policy: Probability distribution over all possible moves
   - Value: Expected game outcome (-1 to +1)

3. **MCTS**: Uses the neural network to guide tree search:
   - Selection: UCB formula balancing exploration/exploitation
   - Expansion: Neural network evaluates new positions
   - Backpropagation: Updates visit counts and values

4. **Training**: The network learns from:
   - Supervised data: Predicting grandmaster moves
   - Self-play: Learning from games against itself
