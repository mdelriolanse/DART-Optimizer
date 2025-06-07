import torch
import torch.nn as nn
from .accelerator import get_accelerator

__all__ = ['MLP']

class MLP(nn.Module):
    def __init__(self, input_size, output_size, device: torch.device=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1000, dtype=torch.float64)
        self.fc2 = nn.Linear(1000, 1000, dtype=torch.float64)
        self.fc3 = nn.Linear(1000, output_size, dtype=torch.float64)
        self.relu = nn.ReLU()

        # Fit model on cpu or available accelerator.
        self.device = get_accelerator()
        self.to(self.device)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x