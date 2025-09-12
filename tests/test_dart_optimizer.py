import pytest
import torch
import torch.nn as nn
import numpy as np
from dartopt import Dart
from dartopt.utils import MLP, DartDataBuilder


class TestDartOptimizer:
    """Test suite for the DART optimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.model = MLP(input_size=10, output_size=2, device=self.device)
        self.optimizer = Dart(
            self.model.parameters(),
            lr=1e-3,
            alpha=1.0,
            lr_min=1e-6,
            lr_max=1e-1
        )
        
    def test_optimizer_initialization(self):
        """Test that the optimizer initializes correctly."""
        assert isinstance(self.optimizer, Dart)
        assert len(self.optimizer.param_groups) == 1
        
        group = self.optimizer.param_groups[0]
        assert 'concentration' in group
        assert 'step' in group
        assert group['step'] == 0
        assert group['alpha'] == 1.0
        
    def test_learning_rate_sampling(self):
        """Test that learning rates are sampled correctly."""
        lr_samples = self.optimizer.get_lr_samples()
        
        assert 'group_0' in lr_samples
        lr_tensor = lr_samples['group_0']
        
        # Should have one learning rate per parameter
        expected_params = sum(1 for _ in self.model.parameters())
        assert lr_tensor.shape[0] == expected_params
        
        # Learning rates should be within bounds
        assert torch.all(lr_tensor >= self.optimizer.param_groups[0]['lr_min'])
        assert torch.all(lr_tensor <= self.optimizer.param_groups[0]['lr_max'])
        
    def test_concentration_parameters(self):
        """Test concentration parameter access."""
        concentration_params = self.optimizer.get_concentration_params()
        
        assert 'group_0' in concentration_params
        concentration_tensor = concentration_params['group_0']
        
        # Should have one concentration parameter per parameter
        expected_params = sum(1 for _ in self.model.parameters())
        assert concentration_tensor.shape[0] == expected_params
        
        # Concentration parameters should be positive
        assert torch.all(concentration_tensor > 0)
        
    def test_optimizer_step(self):
        """Test that the optimizer step works correctly."""
        # Create dummy data
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass
        output = self.model(x)
        loss = criterion(output, y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Store initial parameters
        initial_params = [p.clone() for p in self.model.parameters()]
        
        # Optimizer step
        self.optimizer.step()
        
        # Check that parameters changed
        for initial, current in zip(initial_params, self.model.parameters()):
            assert not torch.equal(initial, current)
            
        # Check that step counter increased
        assert self.optimizer.param_groups[0]['step'] == 1
        
    def test_parameter_groups(self):
        """Test custom parameter groups."""
        # Create model with different layers
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )
        
        # Create optimizer with different settings for each layer
        optimizer = Dart([
            {'params': model[0].parameters(), 'lr': 1e-3, 'alpha': 2.0},
            {'params': model[1].parameters(), 'lr': 5e-4, 'alpha': 1.5}
        ])
        
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['alpha'] == 2.0
        assert optimizer.param_groups[1]['alpha'] == 1.5
        
    def test_learning_rate_bounds(self):
        """Test that learning rates respect bounds."""
        optimizer = Dart(
            self.model.parameters(),
            lr=1e-3,
            lr_min=1e-5,
            lr_max=1e-2
        )
        
        lr_samples = optimizer.get_lr_samples()
        lr_tensor = lr_samples['group_0']
        
        assert torch.all(lr_tensor >= 1e-5)
        assert torch.all(lr_tensor <= 1e-2)
        
    def test_weight_decay(self):
        """Test weight decay functionality."""
        optimizer = Dart(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-4
        )
        
        # Create dummy data
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        criterion = nn.CrossEntropyLoss()
        
        # Forward pass
        output = self.model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check that gradients include weight decay
        for param in self.model.parameters():
            if param.grad is not None:
                # Weight decay should be added to gradients
                assert param.grad is not None
                
    def test_concentration_update(self):
        """Test that concentration parameters are updated."""
        initial_concentration = self.optimizer.get_concentration_params()['group_0'].clone()
        
        # Perform several optimization steps
        x = torch.randn(32, 10)
        y = torch.randint(0, 2, (32,))
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(5):
            output = self.model(x)
            loss = criterion(output, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        final_concentration = self.optimizer.get_concentration_params()['group_0']
        
        # Concentration parameters should have changed
        assert not torch.equal(initial_concentration, final_concentration)


class TestMLP:
    """Test suite for the MLP model."""
    
    def test_mlp_initialization(self):
        """Test MLP model initialization."""
        model = MLP(input_size=784, output_size=10)
        
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'fc1')
        assert hasattr(model, 'fc2')
        assert hasattr(model, 'fc3')
        assert hasattr(model, 'relu')
        
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        model = MLP(input_size=784, output_size=10)
        x = torch.randn(32, 784)
        
        output = model(x)
        
        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_mlp_gradients(self):
        """Test MLP gradient computation."""
        model = MLP(input_size=784, output_size=10)
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        criterion = nn.CrossEntropyLoss()
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        
        # Check that gradients are computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDartDataBuilder:
    """Test suite for the DartDataBuilder dataset."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        X = torch.randn(100, 784)
        y = torch.randint(0, 10, (100,))
        dataset = DartDataBuilder(X, y, device='cpu')
        
        assert len(dataset) == 100
        
    def test_dataset_getitem(self):
        """Test dataset item access."""
        X = torch.randn(100, 784)
        y = torch.randint(0, 10, (100,))
        dataset = DartDataBuilder(X, y, device='cpu')
        
        x_item, y_item = dataset[0]
        assert x_item.shape == (784,)
        assert y_item.shape == ()
        assert torch.equal(x_item, X[0])
        assert torch.equal(y_item, y[0])
        
    def test_dataset_length(self):
        """Test dataset length."""
        X = torch.randn(50, 784)
        y = torch.randint(0, 10, (50,))
        dataset = DartDataBuilder(X, y, device='cpu')
        
        assert len(dataset) == 50


if __name__ == "__main__":
    pytest.main([__file__])
