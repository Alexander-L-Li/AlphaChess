"""
Modal application for supervised chess training.

Trains the neural network on Lichess data before self-play RL.

Usage:
    # Upload dataset to Modal volume
    modal volume put chess-training-volume lichess_dataset_5M.pt /data/

    # Run supervised training
    modal run modal_supervised.py

    # Run in detached mode
    modal run --detach modal_supervised.py

    # Check logs
    modal app logs chess-supervised-training

    # Then run RL training
    modal run --detach modal_app.py
"""

import modal

# Define the Modal app
app = modal.App("chess-supervised-training")

# Use the same volume as RL training
volume = modal.Volume.from_name("chess-training-volume", create_if_missing=True)

# Container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "numpy>=1.24",
        "python-chess>=1.999",
        "tqdm",
    )
    .add_local_file("nn.py", remote_path="/root/nn.py")
    .add_local_file("utils.py", remote_path="/root/utils.py")
    .add_local_file("dataset.py", remote_path="/root/dataset.py")
)

# Volume mount paths
VOLUME_PATH = "/vol"
DATA_PATH = f"{VOLUME_PATH}/data"
MODELS_PATH = f"{VOLUME_PATH}/models"


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,  # 12 hours max
    volumes={VOLUME_PATH: volume},
)
def train_supervised(
    data_file: str = "lichess_dataset_5M.pt",
    output_model: str = "supervised_chess_model.pth",
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    num_workers: int = 4,
    val_split: float = 0.05,
    num_res_blocks: int = 10,
    num_channels: int = 128,
    use_se: bool = True,
):
    """
    Run supervised training on Modal with GPU.

    Args:
        data_file: Name of dataset file in /data/ on volume.
        output_model: Name of output model file.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        num_workers: DataLoader workers.
        val_split: Fraction of data for validation.
        num_res_blocks: Number of residual blocks.
        num_channels: Number of channels.
        use_se: Use Squeeze-and-Excitation blocks.
    """
    import os
    import sys
    sys.path.insert(0, "/root")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from nn import ChessNet
    from utils import ActionMapper, INPUT_PLANES
    from dataset import create_dataset

    # Ensure directories exist
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)

    # Paths
    data_path = f"{DATA_PATH}/{data_file}"
    model_path = f"{MODELS_PATH}/{output_model}"
    checkpoint_path = f"{MODELS_PATH}/supervised_checkpoint.pth"

    print("=" * 60)
    print("Supervised Chess Training on Modal")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"Output: {model_path}")
    print(f"Config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
    print(f"Model: {num_res_blocks} res blocks, {num_channels} channels")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Initialize model
    mapper = ActionMapper()
    model = ChessNet(
        action_size=mapper.vocab_size,
        input_channels=INPUT_PLANES,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
        use_se=use_se
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Check for existing checkpoint
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Checking for compatible checkpoint...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Resumed from epoch {start_epoch}")
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Missing key" in str(e):
                print(f"Checkpoint incompatible with current architecture, starting fresh.")
                start_epoch = 0
            else:
                raise

    # Load dataset
    print(f"\nLoading dataset...")
    dataset = create_dataset(data_path, dataset_type="auto")
    print(f"Dataset size: {len(dataset):,} examples")

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Optimizer and loss functions
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Restore optimizer state if resuming
    if start_epoch > 0 and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Step scheduler to correct position
        for _ in range(start_epoch):
            scheduler.step()

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')

    print("\nStarting training...")
    for epoch in range(start_epoch, epochs):
        # Train
        model.train()
        total_loss = 0
        total_loss_p = 0
        total_loss_v = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y_policy, y_value in pbar:
            x = x.to(device)
            y_policy = y_policy.to(device)
            y_value = y_value.to(device)

            pred_policy_logits, pred_value = model(x)

            loss_p = policy_loss_fn(pred_policy_logits, y_policy)
            loss_v = value_loss_fn(pred_value.squeeze(), y_value)
            loss = loss_p + loss_v

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_loss_p += loss_p.item()
            total_loss_v += loss_v.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'p': f'{loss_p.item():.4f}',
                'v': f'{loss_v.item():.4f}'
            })

        n_batches = len(train_loader)
        train_loss = total_loss / n_batches

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0

        with torch.no_grad():
            for x, y_policy, y_value in val_loader:
                x = x.to(device)
                y_policy = y_policy.to(device)
                y_value = y_value.to(device)

                pred_policy_logits, pred_value = model(x)

                loss_p = policy_loss_fn(pred_policy_logits, y_policy)
                loss_v = value_loss_fn(pred_value.squeeze(), y_value)
                loss = loss_p + loss_v

                val_loss += loss.item() * x.size(0)
                pred_moves = pred_policy_logits.argmax(dim=1)
                val_correct += (pred_moves == y_policy).sum().item()
                val_samples += x.size(0)

        val_loss = val_loss / val_samples
        val_acc = val_correct / val_samples

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2%}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  Saved best model to {model_path}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        torch.save(checkpoint, checkpoint_path)

        # Commit to volume periodically
        if (epoch + 1) % 5 == 0:
            volume.commit()

        print()

    # Final commit
    volume.commit()

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_path}")

    return {
        "status": "complete",
        "best_val_loss": best_val_loss,
        "model_path": model_path
    }


@app.local_entrypoint()
def main(
    data_file: str = "lichess_dataset_5M.pt",
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    num_res_blocks: int = 10,
    num_channels: int = 128,
):
    """
    CLI entrypoint for Modal supervised training.

    Example:
        modal run modal_supervised.py
        modal run modal_supervised.py --epochs 100
        modal run --detach modal_supervised.py
    """
    print("=" * 60)
    print("Launching Supervised Training on Modal")
    print("=" * 60)
    print(f"Dataset: {data_file}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Model: {num_res_blocks} res blocks, {num_channels} channels")
    print()

    result = train_supervised.remote(
        data_file=data_file,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels,
    )

    print(f"\nTraining finished: {result}")
