import torch.nn as nn
import torch.nn.functional as F

class ChessNet(nn.Module):
    def __init__(self, action_size):
        super(ChessNet, self).__init__()
        # Input: 13 planes. Output: 64 filters
        self.conv_input = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
        
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        # Policy Head (Outputting raw move probabilities)
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.Flatten(),
            # Input: 2 channels * 8 * 8 = 128
            # Output: One score for EVERY possible move in the vocabulary
            nn.Linear(128, action_size) 
        )

        # Value Head (Outputting -1 to 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh() # Squeezes output between -1 and 1
        )

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        # Add residual connection
        x = F.relu(x + self.res_block(x))
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value