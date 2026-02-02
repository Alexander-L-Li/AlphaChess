import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_INPUT_CHANNELS = 119 
DEFAULT_RES_BLOCKS = 10
DEFAULT_CHANNELS = 128

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
        out += residual  # skip connection here
        out = F.relu(out)
        return out


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    Used in newer architectures like Leela Chess Zero.
    """
    def __init__(self, channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        # Global average pooling
        y = x.view(batch, channels, -1).mean(dim=2)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch, channels, 1, 1)
        return x * y


class SEResidualBlock(nn.Module):
    """
    Residual block with Squeeze-and-Excitation attention.
    Provides better feature calibration than standard ResBlock.
    """
    def __init__(self, num_channels, se_reduction=4):
        super(SEResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.se = SqueezeExcitation(num_channels, se_reduction)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    def __init__(self, action_size, input_channels=DEFAULT_INPUT_CHANNELS,
                 num_res_blocks=DEFAULT_RES_BLOCKS, num_channels=DEFAULT_CHANNELS,
                 use_se=True):
        """
        CNN architecture inspired by Google DeepMind's AlphaZero.

        Args:
            action_size: Size of policy vector (e.g., mapper.vocab_size).
            input_channels: 119
            num_res_blocks: 19-40 for AlphaZero, 10 default for training efficiency.
            num_channels: 256 for AlphaZero, 128 default for training efficiency.
            use_se: Whether to use Squeeze-and-Excitation blocks (like Leela Chess Zero).
        """
        super(ChessNet, self).__init__()

        self.input_channels = input_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks

        # Input conv block
        self.conv_input = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower
        if use_se:
            self.res_tower = nn.Sequential(
                *[SEResidualBlock(num_channels) for _ in range(num_res_blocks)]
            )
        else:
            self.res_tower = nn.Sequential(
                *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
            )

        # Policy head (slightly larger)
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, action_size)

        # Value head (slightly larger)
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
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
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def count_parameters(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ChessNetSmall(ChessNet):
    """Smaller network for faster iteration during development."""
    def __init__(self, action_size):
        super().__init__(
            action_size,
            input_channels=119,
            num_res_blocks=6,
            num_channels=64,
            use_se=False
        )