import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    The standard residual block, inspired by Google DeepMind AlphaZero.
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> Add Skip -> ReLU
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # skip connection here
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    def __init__(self, action_size, input_channels=13, num_res_blocks=4, num_channels=64):
        """
        Args:
            action_size: Size of policy vector (e.g., mapper.vocab_size).
            input_channels: 119 for full AlphaZero[cite: 469], 13 for your simple version.
            num_res_blocks: 19 for standard AlphaZero, 4-6 for MacBook.
            num_channels: 256 for standard AlphaZero, 64 for MacBook.
        """
        super(ChessNet, self).__init__()
        
        # --- Initial Convolutional Block ---
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # --- Residual Tower ---
        self.res_tower = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        # Flatten size: 2 filters * 8 * 8 board = 128 features
        self.policy_fc = nn.Linear(2 * 8 * 8, action_size)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        # Flatten size: 1 filter * 8 * 8 = 64 features
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # 1. Initial Block
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # 2. Residual Tower
        x = self.res_tower(x)
        
        # 3. Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten
        p = self.policy_fc(p)
        
        # 4. Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v