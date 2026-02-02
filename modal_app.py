"""
Modal application for AlphaZero-style chess training.

Deploy training to Modal cloud with GPU acceleration.

Usage:
    # Setup (first time only)
    pip install modal
    modal setup

    # Create volume and upload pre-trained model
    modal volume create chess-training-volume
    modal volume put chess-training-volume supervised_chess_model.pth /models/supervised_chess_model.pth

    # Run training (default: 100 iterations, 100 games/iter)
    modal run modal_app.py

    # Run with custom parameters
    modal run modal_app.py --iterations 50 --games-per-iter 200

    # Run in detached mode (continues after terminal closes)
    modal run --detach modal_app.py

    # Monitor logs
    modal app logs alphazero-chess-training

    # Download trained model
    modal volume get chess-training-volume /models/rl_chess_model_best.pth ./
"""

import modal

# Define the Modal app
app = modal.App("alphazero-chess-training")

# Create a persistent volume for models and checkpoints
volume = modal.Volume.from_name("chess-training-volume", create_if_missing=True)

# Define the container image with dependencies and local source files
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "python-chess>=1.999",
        "tqdm",
    )
    .add_local_file("nn.py", remote_path="/root/nn.py")
    .add_local_file("mcts.py", remote_path="/root/mcts.py")
    .add_local_file("utils.py", remote_path="/root/utils.py")
    .add_local_file("modal_train.py", remote_path="/root/modal_train.py")
)

# Volume mount paths
VOLUME_PATH = "/vol"
MODELS_PATH = f"{VOLUME_PATH}/models"
CHECKPOINTS_PATH = f"{VOLUME_PATH}/checkpoints"


@app.function(
    image=image,
    gpu="A10G",  # Good balance of performance and cost ($1.10/hr)
    timeout=60 * 60 * 24,  # 24 hours max (Modal limit)
    volumes={VOLUME_PATH: volume},
)
def train_alphazero(
    # Training iterations
    iterations: int = 100,
    games_per_iter: int = 500,
    mcts_sims: int = 400,
    batch_size: int = 256,
    epochs: int = 50,
    max_buffer_size: int = 500000,
    mcts_batch_size: int = 32,
    cache_size: int = 100000,
    # Model architecture
    num_res_blocks: int = 10,
    num_channels: int = 128,
    use_se: bool = True,
    # Learning rate schedule
    initial_lr: float = 0.01,
    # Arena settings
    arena_games: int = 100,
    win_threshold: float = 0.57,
    arena_interval: int = 1,
    # Runtime
    max_runtime_hours: float = 23.0,
):
    """
    Run AlphaZero training on Modal with GPU.

    This function runs the complete AlphaZero training loop:
    1. Self-play: Generate games using MCTS
    2. Training: Update neural network on game data
    3. Arena: Compare new model against previous best

    Args:
        iterations: Number of self-play/training iterations.
        games_per_iter: Number of games per iteration.
        mcts_sims: MCTS simulations per move during self-play.
        batch_size: Training batch size.
        epochs: Training epochs per iteration.
        max_buffer_size: Maximum replay buffer size.
        mcts_batch_size: Batch size for MCTS neural net evaluation.
        cache_size: Transposition table size.
        num_res_blocks: Number of residual blocks in the model.
        num_channels: Number of channels in the model.
        use_se: Whether to use Squeeze-and-Excitation blocks.
        initial_lr: Initial learning rate.
        arena_games: Number of games in arena evaluation.
        win_threshold: Win rate threshold to accept new model.
        arena_interval: Run arena every N iterations.
        max_runtime_hours: Exit gracefully before this time.
    """
    import os
    import sys

    # Add mount directory to path
    sys.path.insert(0, "/root")

    # Ensure directories exist
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

    # Import training module
    from modal_train import run_training

    # Define paths on the volume
    supervised_model = f"{MODELS_PATH}/supervised_chess_model.pth"
    model_output = f"{MODELS_PATH}/rl_chess_model_latest.pth"
    best_model = f"{MODELS_PATH}/rl_chess_model_best.pth"
    checkpoint = f"{CHECKPOINTS_PATH}/rl_checkpoint.pth"

    print(f"=" * 60)
    print("AlphaZero Chess Training on Modal")
    print(f"=" * 60)
    print(f"Supervised baseline: {supervised_model}")
    print(f"Model output: {model_output}")
    print(f"Best model: {best_model}")
    print(f"Checkpoint: {checkpoint}")
    print()

    # Build config dictionary
    config = {
        # Training iterations
        "iterations": iterations,
        "games_per_iter": games_per_iter,
        "mcts_sims": mcts_sims,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_buffer_size": max_buffer_size,
        "mcts_batch_size": mcts_batch_size,
        "cache_size": cache_size,
        # Model architecture
        "num_res_blocks": num_res_blocks,
        "num_channels": num_channels,
        "use_se": use_se,
        # Learning rate schedule
        "initial_lr": initial_lr,
        "lr_drops": {30: 0.002, 60: 0.0002, 80: 0.00002},
        # Arena settings
        "arena_games": arena_games,
        "win_threshold": win_threshold,
        "arena_interval": arena_interval,
        # Temperature settings
        "temp_threshold": 30,
        "temp_final": 0.1,
        # Gradient clipping
        "max_grad_norm": 1.0,
        # Runtime limit
        "max_runtime_hours": max_runtime_hours,
    }

    # Run training
    run_training(
        supervised_model_path=supervised_model,
        model_output_path=model_output,
        best_model_path=best_model,
        checkpoint_path=checkpoint,
        config=config,
    )

    # Commit volume changes
    volume.commit()

    print("Training complete! Models saved to volume.")
    return {
        "status": "complete",
        "model_path": model_output,
        "best_model_path": best_model,
    }


@app.local_entrypoint()
def main(
    iterations: int = 100,
    games_per_iter: int = 500,
    mcts_sims: int = 400,
    batch_size: int = 256,
    epochs: int = 50,
    num_res_blocks: int = 10,
    num_channels: int = 128,
    arena_games: int = 100,
    max_runtime_hours: float = 23.0,
):
    """
    CLI entrypoint for Modal training.

    Example:
        modal run modal_app.py
        modal run modal_app.py --iterations 50 --games-per-iter 200
        modal run --detach modal_app.py  # Run in background

    Training auto-saves checkpoints and exits gracefully before 24hr timeout.
    Restart to continue from checkpoint: modal run --detach modal_app.py
    """
    print("=" * 60)
    print("Launching AlphaZero Chess Training on Modal")
    print("=" * 60)
    print(f"Iterations: {iterations}")
    print(f"Games per iteration: {games_per_iter}")
    print(f"MCTS simulations: {mcts_sims}")
    print(f"Model: {num_res_blocks} res blocks, {num_channels} channels")
    print(f"Arena: {arena_games} games every iteration (vs supervised baseline)")
    print(f"Max runtime: {max_runtime_hours} hours")
    print()

    result = train_alphazero.remote(
        iterations=iterations,
        games_per_iter=games_per_iter,
        mcts_sims=mcts_sims,
        batch_size=batch_size,
        epochs=epochs,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        arena_games=arena_games,
        max_runtime_hours=max_runtime_hours,
    )

    print(f"\nTraining finished: {result}")
