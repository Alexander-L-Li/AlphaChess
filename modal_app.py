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

    # Run training
    modal run modal_app.py

    # Run with custom parameters
    modal run modal_app.py --iterations 5 --games-per-iter 10

    # Monitor logs
    modal app logs alphazero-chess-training

    # Download trained model
    modal volume get chess-training-volume /models/rl_chess_model_latest.pth ./
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
    gpu="T4",  # Budget option, $0.59/hr - best value for long runs
    timeout=60 * 60 * 24,  # 24 hours max (Modal limit)
    volumes={VOLUME_PATH: volume},
)
def train_alphazero(
    iterations: int = 10,
    games_per_iter: int = 50,
    mcts_sims: int = 800,
    batch_size: int = 256,
    epochs: int = 10,
    max_buffer_size: int = 100000,
    mcts_batch_size: int = 64,
    cache_size: int = 50000,
    max_runtime_hours: float = 23.0,  # Exit gracefully before 24hr Modal timeout
):
    """
    Run AlphaZero training on Modal with GPU.

    Args:
        iterations: Number of self-play/training iterations.
        games_per_iter: Number of games per iteration.
        mcts_sims: MCTS simulations per move.
        batch_size: Training batch size.
        epochs: Training epochs per iteration.
        max_buffer_size: Maximum replay buffer size.
        mcts_batch_size: Batch size for MCTS evaluation.
        cache_size: Transposition table size.
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
    model_input = f"{MODELS_PATH}/supervised_chess_model.pth"
    model_output = f"{MODELS_PATH}/rl_chess_model_latest.pth"
    checkpoint = f"{CHECKPOINTS_PATH}/rl_checkpoint.pth"

    print(f"Starting AlphaZero training on Modal")
    print(f"  Model input: {model_input}")
    print(f"  Model output: {model_output}")
    print(f"  Checkpoint: {checkpoint}")

    # Run training
    run_training(
        model_input_path=model_input,
        model_output_path=model_output,
        checkpoint_path=checkpoint,
        iterations=iterations,
        games_per_iter=games_per_iter,
        mcts_sims=mcts_sims,
        batch_size=batch_size,
        epochs=epochs,
        max_buffer_size=max_buffer_size,
        mcts_batch_size=mcts_batch_size,
        cache_size=cache_size,
        max_runtime_hours=max_runtime_hours,
    )

    # Commit volume changes
    volume.commit()

    print("Training complete! Model saved to volume.")
    return {"status": "complete", "model_path": model_output}


@app.local_entrypoint()
def main(
    iterations: int = 10,
    games_per_iter: int = 50,
    mcts_sims: int = 800,
    batch_size: int = 256,
    epochs: int = 10,
    max_buffer_size: int = 100000,
    mcts_batch_size: int = 64,
    cache_size: int = 50000,
    max_runtime_hours: float = 23.0,
):
    """
    CLI entrypoint for Modal training.

    Example:
        modal run modal_app.py --iterations 5 --games-per-iter 10

    Training auto-saves checkpoints and exits gracefully before 24hr timeout.
    Just restart to continue: modal run --detach modal_app.py
    """
    print("Launching AlphaZero training on Modal...")
    print(f"  Iterations: {iterations}")
    print(f"  Games per iteration: {games_per_iter}")
    print(f"  MCTS simulations: {mcts_sims}")
    print(f"  Max runtime: {max_runtime_hours} hours (will checkpoint and exit before timeout)")

    result = train_alphazero.remote(
        iterations=iterations,
        games_per_iter=games_per_iter,
        mcts_sims=mcts_sims,
        batch_size=batch_size,
        epochs=epochs,
        max_buffer_size=max_buffer_size,
        mcts_batch_size=mcts_batch_size,
        cache_size=cache_size,
        max_runtime_hours=max_runtime_hours,
    )

    print(f"Training finished: {result}")
