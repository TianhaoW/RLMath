# --- Torch / Device checks for macOS (Apple Silicon M4) ---
import torch
import torch.nn as nn
import torch.nn.functional as F

def device_selector():

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())  # Usually False on Apple Silicon

    # Metal (MPS) backend (Apple GPU)
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    mps_built = hasattr(torch.backends, "mps") and torch.backends.mps.is_built()

    print("MPS built:", mps_built)
    print("MPS available:", mps_available)

    device = torch.device(
        "mps" if mps_available else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Using device:", device)

    # Optional quick sanity test on selected device
    try:
        x = torch.randn(2, 2, device=device)
        print("Test tensor sum:", x.sum().item())
    except Exception as e:
        print("Device test failed:", e)

    torch.manual_seed(0)

    return device

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device=device_selector()):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(2, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )

        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x