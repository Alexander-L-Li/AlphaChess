import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    The standard residual block, inspired by Google DeepMind AlphaZero.
    Structure: Conv -> BN -> ReLU -> Conv -> BN -> Add Skip -> ReLU
    """
    def __init__(self, num_channels):
        """
        Initialize a residual block.

        Args:
            num_channels: Number of input and output channels.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x: Input tensor of shape (batch, channels, 8, 8).

        Returns:
            Output tensor of the same shape with residual connection applied.
        """
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # skip connection here
        out = F.relu(out)
        return out

class ChessNet(nn.Module):
    def __init__(self, action_size, input_channels=13, num_res_blocks=4, num_channels=64):
        """
        CNN architecture inspired by Google DeepMind's AlphaZero
        Args:
            action_size: Size of policy vector (e.g., mapper.vocab_size).
            input_channels: 119 for full AlphaZero, 13 for MacBook.
            num_res_blocks: 19 for standard AlphaZero, 4-6 for MacBook.
            num_channels: 256 for standard AlphaZero, 64 for MacBook.
        """
        super(ChessNet, self).__init__()
        
        # Conv block
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Res tower
        self.res_tower = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, input_channels, 8, 8).

        Returns:
            Tuple of (policy_logits, value) where policy_logits has shape
            (batch, action_size) and value has shape (batch, 1).
        """
        # Initial block
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Res tower
        x = self.res_tower(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # Flatten
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v