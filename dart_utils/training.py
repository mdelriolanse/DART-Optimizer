import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from typing import Callable


# Train the Model
class Trainer():
    def __init__(self, data, num_epochs: int = None, device: torch.device = 'cpu') -> None:
        print(f"Fitting training on {device}.")
        self.data = data
        self.epochs = 30 if num_epochs is None else num_epochs
        
    def store_training_history(self, history: dict, epoch_num: int, loss: list, accuracy: list) -> dict:
        detacher = lambda x: torch.tensor(x).detach().cpu().numpy()

        """
        This function stores training metrics in a dictionary.
        
        Args:
            history: A dictionary to store training metrics (loss, accuracy, etc.).
            epoch: The current epoch number (int).
            loss: The training loss value (float).
            accuracy: The training accuracy value (float).
            other_metrics: A dictionary containing additional metrics to store (optional).
        """
        
        history[epoch_num] = {}
        history[epoch_num]['loss'] = np.mean(detacher(loss))
        # history[epoch_num]['accuracy'] = np.mean(detacher(accuracy))
        
    def train(self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: Callable[[Tensor, Tensor], Tensor]) -> dict:
        '''
        Train Model Given Loss Function and Optimizer
        '''
        model.train()
        training_history = {}

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

                optimizer.zero_grad() # reset gradients
                loss.backward() # compute gradients for all 128 samples

                optimizer.step() # apply weight update and pass loss

                losses.append(loss.to('cpu'))
            self.store_training_history(history=training_history,
                                   epoch_num=epoch,
                                   loss=losses,
                                   accuracy=accuracies)
            with torch.no_grad():
                print(f"Completed Epoch: {epoch+1}/{self.epochs}, Loss: {np.mean(losses):.4f}")
        return training_history