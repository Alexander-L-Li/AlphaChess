# AlphaChess

An AlphaZero-style chess engine built from scratch using PyTorch. The engine combines a convolutional neural network with Monte Carlo Tree Search (MCTS) to play chess. The original AlphaZero by Google Deepmind performed approximately 44 million games of self-play during its training for chess. It accomplished this in just 4 hours using 5,000 first-generation TPUs.

## Features

- **Neural Network**: ResNet-style architecture with separate policy and value heads
- **MCTS**: Batched Monte Carlo Tree Search with UCB selection, Dirichlet noise, and transposition tables
- **Training Pipeline**:
  - Supervised learning on Lichess game data
  - Self-play reinforcement learning (AlphaZero approach)
  - **Cloud training on Modal** with GPU acceleration
- **Interactive GUI**: Pygame-based interface to play against the engine

## Project Structure

```
├── nn.py              # Neural network architecture (ChessNet)
├── mcts.py            # Batched MCTS with caching and virtual loss
├── train.py           # Supervised training on Lichess data
├── alpha_train.py     # Local self-play reinforcement learning
├── modal_train.py     # Modal-compatible training script
├── modal_app.py       # Modal cloud deployment configuration
├── play.py            # Pygame GUI to play against the engine
├── utils.py           # Board encoding and action mapping utilities
├── dataset.py         # Dataset loader for training data
├── eval.py            # Evaluation against Stockfish
├── requirements-modal.txt  # Minimal dependencies for cloud training
└── assets/            # Chess piece images for GUI
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

2. **Self-Play Reinforcement Learning** (local):

   ```bash
   python alpha_train.py
   ```

3. **Cloud Training on Modal** (recommended for full training):

   ```bash
   # Install Modal CLI
   pip install modal
   modal setup

   # Create volume and upload pre-trained model
   modal volume create chess-training-volume
   modal volume put chess-training-volume supervised_chess_model.pth /models/supervised_chess_model.pth

   # Run training (detached - survives disconnection)
   modal run --detach modal_app.py

   # Monitor progress
   modal app list

   # Download trained model when complete
   modal volume get chess-training-volume /models/rl_chess_model_latest.pth ./
   ```

   **Cloud Training Config** (default):
   - GPU: NVIDIA T4 ($0.59/hr)
   - MCTS simulations: 800 (AlphaZero paper spec)
   - Games: 10 iterations x 50 games = 500 games
   - Auto-checkpoints every iteration
   - Graceful exit at 23hrs (Modal 24hr limit) - just restart to continue

### Playing Against the Engine

```bash
python play.py
```

Click to select a piece, then click the destination square to move. The AI will respond automatically.

### Evaluation

Evaluate the model against Stockfish:

```bash
python eval.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- python-chess
- pygame (for GUI)
- numpy
- tqdm

## How It Works

1. **Board Representation**: The board is encoded as a 13-channel 8x8 tensor (12 piece types + turn indicator)

2. **Neural Network**: Takes the board state and outputs:
   - Policy: Probability distribution over all possible moves (~4000 possible moves)
   - Value: Expected game outcome (-1 to +1)

3. **Batched MCTS**: Optimized tree search with:
   - Selection: UCB formula with virtual loss for parallel evaluation
   - Batched expansion: Multiple positions evaluated in single GPU call
   - Transposition table: LRU cache to avoid re-evaluating positions
   - Early stopping: Exits when dominant move emerges
   - Subtree reuse: Preserves search tree between moves

4. **Training Pipeline**:
   - Supervised pre-training on grandmaster games
   - Self-play reinforcement learning with experience replay
   - Checkpoint resumption for long training runs
   - Multi-device support: CUDA > MPS > CPU

## Cloud Training Architecture

The Modal deployment (`modal_app.py`) provides:

- Persistent volume for models and checkpoints
- Automatic checkpoint saves after each iteration
- Graceful timeout handling (exits before 24hr Modal limit)
- Resume from checkpoint on restart

```
/vol/
├── models/
│   ├── supervised_chess_model.pth    # Upload before training
│   └── rl_chess_model_latest.pth     # Output after training
└── checkpoints/
    └── rl_checkpoint.pth             # For resumption
```
