import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from torch import Tensor
import math

class Dart(torch.optim.Optimizer):
    """
    DART: Dirichlet Adaptive Random Tuning Optimizer
    
    A novel optimization algorithm that combines Bayesian inference, adaptive learning rates,
    and momentum-based methods by sampling learning rates from a Dirichlet distribution.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Base learning rate (default: 1e-3)
        betas: Coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps: Term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 0)
        alpha: Dirichlet concentration parameter (default: 1.0)
        lr_min: Minimum learning rate for clamping (default: 1e-6)
        lr_max: Maximum learning rate for clamping (default: 1e-1)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        alpha: float = 1.0,
        lr_min: float = 1e-6,
        lr_max: float = 1e-1,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 < alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 < lr_min < lr_max:
            raise ValueError(f"Invalid learning rate bounds: {lr_min} < {lr_max}")
            
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            alpha=alpha,
            lr_min=lr_min,
            lr_max=lr_max,
        )
        super(Dart, self).__init__(params, defaults)
        
        # Initialize Dirichlet concentration parameters
        for group in self.param_groups:
            group['concentration'] = torch.full((len(group['params']),), group['alpha'], 
                                              dtype=torch.float32, device=group['params'][0].device)
            group['step'] = 0

    def _sample_learning_rates(self, group: Dict[str, Any]) -> Tensor:
        """
        Sample learning rates from Dirichlet distribution.
        
        For each parameter tensor, we sample candidate learning rates from a normal distribution
        and then apply Dirichlet reparameterization.
        """
        concentration = group['concentration']
        lr_base = group['lr']
        lr_min, lr_max = group['lr_min'], group['lr_max']
        
        # Sample from Dirichlet distribution using reparameterization trick
        # Sample from Gamma distributions and normalize
        gamma_samples = torch.distributions.Gamma(concentration, torch.ones_like(concentration)).sample()
        dirichlet_samples = gamma_samples / gamma_samples.sum()
        
        # Scale to learning rate range
        lr_candidates = lr_min + (lr_max - lr_min) * dirichlet_samples
        
        return lr_candidates

    def _update_concentration(self, group: Dict[str, Any], gradients: Tensor, lr_samples: Tensor):
        """
        Update Dirichlet concentration parameters based on gradient information.
        
        This implements the adaptive moment estimation inspired by Adam,
        but applied to the concentration parameters of the Dirichlet distribution.
        """
        beta1, beta2 = group['betas']
        eps = group['eps']
        
        # Initialize moment estimates if not present
        if 'exp_avg' not in group:
            group['exp_avg'] = torch.zeros_like(group['concentration'])
            group['exp_avg_sq'] = torch.zeros_like(group['concentration'])
        
        # Compute gradients w.r.t. concentration parameters
        # This is a simplified approximation - in practice, you'd need proper
        # reparameterization gradients through the Dirichlet sampling
        concentration_grads = gradients.mean(dim=0) if gradients.dim() > 0 else gradients
        
        # Update biased first moment estimate
        group['exp_avg'] = beta1 * group['exp_avg'] + (1 - beta1) * concentration_grads
        
        # Update biased second raw moment estimate
        group['exp_avg_sq'] = beta2 * group['exp_avg_sq'] + (1 - beta2) * concentration_grads.pow(2)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** group['step']
        bias_correction2 = 1 - beta2 ** group['step']
        
        # Update concentration parameters
        denom = (group['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        concentration_update = group['exp_avg'] / bias_correction1 / denom
        
        # Apply update with learning rate
        group['concentration'] = torch.clamp(
            group['concentration'] + 0.01 * concentration_update,
            min=0.1, max=10.0  # Keep concentration parameters in reasonable range
        )

    def step(self, layer_gradients: Optional[Dict] = None, closure: Optional[callable] = None):
        """
        Performs a single optimization step.
        
        Args:
            layer_gradients: Dictionary containing layer-wise gradient information
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group['step'] += 1
            
            # Sample learning rates for this step
            lr_samples = self._sample_learning_rates(group)
            
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('DART does not support sparse gradients')
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Get sampled learning rate for this parameter
                lr = lr_samples[i].item()
                
                # Simple gradient descent step with sampled learning rate
                p.data.add_(grad, alpha=-lr)
                
                # Update concentration parameters based on gradient information
                if layer_gradients is not None and i < len(lr_samples):
                    self._update_concentration(group, grad, lr_samples[i])

        return loss

    def get_lr_samples(self) -> Dict[str, Tensor]:
        """
        Get the current sampled learning rates for each parameter group.
        
        Returns:
            Dictionary mapping group names to tensors of sampled learning rates
        """
        lr_samples = {}
        for i, group in enumerate(self.param_groups):
            lr_samples[f'group_{i}'] = self._sample_learning_rates(group)
        return lr_samples

    def get_concentration_params(self) -> Dict[str, Tensor]:
        """
        Get the current Dirichlet concentration parameters.
        
        Returns:
            Dictionary mapping group names to concentration parameter tensors
        """
        concentration_params = {}
        for i, group in enumerate(self.param_groups):
            concentration_params[f'group_{i}'] = group['concentration'].clone()
        return concentration_params
