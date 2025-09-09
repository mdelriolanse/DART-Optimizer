import torch
import torch.nn as nn
from dartopt.utils.accelerator import get_accelerator

__all__ = ['MLP']

class MLP(nn.Module):
    def __init__(self, input_size, output_size, device: torch.device=None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1000, dtype=torch.float64)
        self.fc2 = nn.Linear(1000, 1000, dtype=torch.float64)
        self.fc3 = nn.Linear(1000, output_size, dtype=torch.float64)
        self.relu = nn.ReLU()

        # Fit model on cpu or available accelerator.
        if device is None:
            self.device = get_accelerator()
        else:
            self.device = device
        self.to(self.device)

    def _compile_model_grads(self):
        modules_grads = {}
        for layer in self.__dict__['_modules'].values():
            if hasattr(layer, "weight"):
                modules_grads[layer] = {"params" : layer.weight, "grads" : layer.weight.grad}

        return modules_grads

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
