import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nn import ChessNet
from utils import ActionMapper
from dataset import ChessDataset

# 1. Setup
mapper = ActionMapper()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ChessNet(action_size=mapper.vocab_size).to(device)
dataset = ChessDataset("lichess_dataset_10k.pt")
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
epoch_range = 100

# 2. Optimizers
optimizer = optim.Adam(model.parameters(), lr=0.001)
policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn = nn.MSELoss()

# 3. Loop
model.train()
for epoch in range(epoch_range):
    total_loss = 0
    
    for batch_idx, (x, y_policy, y_value) in enumerate(dataloader):
        x, y_policy, y_value = x.to(device), y_policy.to(device), y_value.to(device)
        
        # Forward
        pred_policy_logits, pred_value = model(x)
        
        # Calculate Loss
        # Policy: "Did we guess the move the GM played?"
        loss_p = policy_loss_fn(pred_policy_logits, y_policy)
        
        # Value: "Did we predict who won?" (pred_value is (B, 1), target is (B))
        loss_v = value_loss_fn(pred_value.squeeze(), y_value)
        
        loss = loss_p + loss_v
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), "supervised_chess_model.pth")