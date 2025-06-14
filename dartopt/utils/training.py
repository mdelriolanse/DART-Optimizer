import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from typing import Callable, Union
from torch.utils.data import Dataset
from collections import defaultdict

__all__ = ['Trainer', 'DartDataBuilder']

class DartDataBuilder(Dataset):
    def __init__(self, X, Y, device: Union[torch.device, str]):
        self.x = X.to(device)
        self.y = Y.to(device)
        self.len = torch.squeeze(self.x).shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len

# Train the Model
class Trainer():
    def __init__(self, data, num_epochs: int = None, device: torch.device = 'cpu') -> None:
        print(f"Fitting training on {device}.")
        self.data = data
        self.epochs = 30 if num_epochs is None else num_epochs
        
    def store_training_history(self, history: dict, epoch_num: int, loss: list, accuracy: list) -> dict:
        detacher = lambda x: torch.tensor(x).detach().cpu().numpy()

        """
        This function stores loss metrics in a dictionary.
        
        Args:
            history: A dictionary to store training metrics (loss, accuracy, etc.).
            epoch: The current epoch number (int).
            loss: The training loss value (float).
        """
        
        history[epoch_num]['loss'] = np.mean(detacher(loss))
        
    def train(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]) -> dict:
        '''
        Train Model Given Loss Function and Optimizer
        '''

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not a valid PyTorch optimizer') 
        model.train()
        training_history = defaultdict(dict)
        
        for epoch in range(self.epochs):
            losses = []
            accuracies = []
            for images, labels in self.data:
                images = torch.squeeze(images)
                labels = torch.squeeze(labels)

                assert images.shape == torch.Size([128, 784]), f"Images [{images.shape}] not of desired dimensions."
                assert labels.shape == torch.Size([128]), f"Labels [{labels.shape}] not of desired dimensions."
                
                
                predictions = model(images)

                loss = criterion(predictions, labels)
    
                optimizer.zero_grad(set_to_none=True) # reset gradients
                loss.backward(retain_graph=True) # compute gradients for all 128 samples

                layer_gradients = model._compile_model_grads()
                
                optimizer.step(layer_gradients) # apply weight update and pass loss
    
                losses.append(loss.to('cpu'))
            self.store_training_history(history=training_history,
                                       epoch_num=epoch,
                                       loss=losses,
                                       accuracy=accuracies)
            with torch.no_grad():
                print(f"Completed Epoch: {epoch+1}/{self.epochs}, Loss: {np.mean(losses):.4f}")
        return training_history
