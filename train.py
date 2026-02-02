"""
Supervised training script for chess model.

Trains the neural network on Lichess game data before self-play reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn import ChessNet, DEFAULT_RES_BLOCKS, DEFAULT_CHANNELS
from utils import ActionMapper, INPUT_PLANES
from dataset import create_dataset

# =============================================================================
# CONFIGURATION - Edit these values
# =============================================================================

DATA_FILE = "lichess_dataset_5M.pt"
OUTPUT_MODEL = "supervised_chess_model.pth"
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_WORKERS = 4
VAL_SPLIT = 0.05

# Model architecture (should match modal_train.py)
NUM_RES_BLOCKS = DEFAULT_RES_BLOCKS  # 10
NUM_CHANNELS = DEFAULT_CHANNELS      # 128
USE_SE = True

# =============================================================================


def get_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(model, dataloader, optimizer, policy_loss_fn, value_loss_fn, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_loss_p = 0
    total_loss_v = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

    for x, y_policy, y_value in pbar:
        x = x.to(device)
        y_policy = y_policy.to(device)
        y_value = y_value.to(device)

        # Forward pass
        pred_policy_logits, pred_value = model(x)

        # Calculate losses
        loss_p = policy_loss_fn(pred_policy_logits, y_policy)
        loss_v = value_loss_fn(pred_value.squeeze(), y_value)
        loss = loss_p + loss_v

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_loss_p += loss_p.item()
        total_loss_v += loss_v.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'p': f'{loss_p.item():.4f}',
            'v': f'{loss_v.item():.4f}'
        })

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'policy_loss': total_loss_p / n_batches,
        'value_loss': total_loss_v / n_batches
    }


def validate(model, dataloader, policy_loss_fn, value_loss_fn, device):
    """Run validation."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y_policy, y_value in dataloader:
            x = x.to(device)
            y_policy = y_policy.to(device)
            y_value = y_value.to(device)

            pred_policy_logits, pred_value = model(x)

            loss_p = policy_loss_fn(pred_policy_logits, y_policy)
            loss_v = value_loss_fn(pred_value.squeeze(), y_value)
            loss = loss_p + loss_v

            total_loss += loss.item() * x.size(0)

            # Policy accuracy (top-1)
            pred_moves = pred_policy_logits.argmax(dim=1)
            total_correct += (pred_moves == y_policy).sum().item()
            total_samples += x.size(0)

    return {
        'loss': total_loss / total_samples,
        'accuracy': total_correct / total_samples
    }


def main():
    device = get_device()
    print(f"Supervised Training on {device}")
    print(f"Config: epochs={EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")
    print(f"Model: {NUM_RES_BLOCKS} res blocks, {NUM_CHANNELS} channels, {INPUT_PLANES} input planes")
    print()

    # Initialize model
    mapper = ActionMapper()
    model = ChessNet(
        action_size=mapper.vocab_size,
        input_channels=INPUT_PLANES,
        num_res_blocks=NUM_RES_BLOCKS,
        num_channels=NUM_CHANNELS,
        use_se=USE_SE
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Load dataset
    print(f"\nLoading dataset from {DATA_FILE}...")
    dataset = create_dataset(DATA_FILE, dataset_type="auto")
    print(f"Dataset size: {len(dataset):,} examples")

    # Split into train/val
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Optimizer and loss functions
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    # Training loop
    best_val_loss = float('inf')

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer,
            policy_loss_fn, value_loss_fn, device, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, policy_loss_fn, value_loss_fn, device)

        # Step scheduler
        scheduler.step()

        # Log metrics
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} "
              f"(p={train_metrics['policy_loss']:.4f}, v={train_metrics['value_loss']:.4f})")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Accuracy: {val_metrics['accuracy']:.2%}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), OUTPUT_MODEL)
            print(f"  Saved best model to {OUTPUT_MODEL}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss']
        }
        torch.save(checkpoint, "supervised_checkpoint.pth")

        print()

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
