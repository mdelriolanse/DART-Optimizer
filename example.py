#!/usr/bin/env python3
"""
Example usage of the DART optimizer on a simple classification task.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dartopt import Dart
from dartopt.utils import MLP, Trainer, DartDataBuilder


def main():
    """Run a simple example with DART optimizer."""
    print("DART Optimizer Example")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic data (simulating MNIST-like data)
    print("\nCreating synthetic dataset...")
    X = torch.randn(1000, 784)  # 1000 samples, 784 features
    y = torch.randint(0, 10, (1000,))  # 10 classes
    
    # Create dataset and dataloader
    dataset = DartDataBuilder(X, y, device=device)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Create model
    print("Creating MLP model...")
    model = MLP(input_size=784, output_size=10, device=device)
    
    # Create DART optimizer
    print("Initializing DART optimizer...")
    optimizer = Dart(
        model.parameters(),
        lr=1e-3,           # Base learning rate
        alpha=1.0,         # Dirichlet concentration parameter
        lr_min=1e-6,       # Minimum learning rate
        lr_max=1e-1,       # Maximum learning rate
        betas=(0.9, 0.999), # Adam-style momentum parameters
        weight_decay=1e-4   # L2 regularization
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(dataloader, num_epochs=10, device=device)
    
    # Train the model
    print("\nStarting training...")
    print("-" * 30)
    history = trainer.train(model, optimizer, criterion)
    
    # Display results
    print("\nTraining completed!")
    print("=" * 50)
    
    # Show learning rate samples
    lr_samples = optimizer.get_lr_samples()
    concentration_params = optimizer.get_concentration_params()
    
    print(f"Final learning rate samples: {lr_samples['group_0']}")
    print(f"Final concentration parameters: {concentration_params['group_0']}")
    
    # Test the model
    print("\nTesting model...")
    model.eval()
    with torch.no_grad():
        test_x = torch.randn(100, 784).to(device)
        test_y = torch.randint(0, 10, (100,)).to(device)
        
        predictions = model(test_x)
        predicted_classes = torch.argmax(predictions, dim=1)
        accuracy = (predicted_classes == test_y).float().mean()
        
        print(f"Test accuracy: {accuracy.item():.4f}")


if __name__ == "__main__":
    main()
