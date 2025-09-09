# DART: Dirichlet Adaptive Random Tuning Optimizer
DART is a novel optimization algorithm designed for deep learning, combining the strengths of Bayesian inference, adaptive learning rates, and momentum-based methods. It advances beyond static or cyclic schedules by integrating a probabilistic learning rate framework, sampling directly from a Dirichlet distribution to promote dynamic, multimodal learning behavior.


## Key Features
- Dirichlet-distributed learning rates enabling multimodal, adaptive exploration.
- Implicit reparameterization gradients for backpropagation through sampling.
- Adaptive moment estimation inspired by Adam, using gradients w.r.t. Dirichlet concentration parameters.
- Bias correction to ensure stable updates early in training.
- Improved convergence and interpretability in stochastic environments.
## Theoretical Foundation

This work extends findings by:
- Loschitov & Hutter on warm restarts in SGD,
- An et al. and Yu et al. on cyclic learning rate schedules,
- Kingma & Welling on variational inference,
- Joo et al. on the Dirichlet Variational Autoencoder (DirVAE).

**Core Concept:**
DART replaces fixed or hand-tuned learning rate schedules with learnable probabilistic distributions. At each step, candidate learning rates are sampled from a Dirichlet distribution, and adjusted based on backpropagated gradient information.

## Mathematical Formulation

### 1. Learning Rate Sampling

For each parameter tensor, we sample candidate learning rates from a Dirichlet distribution using the reparameterization trick:

**Step 1: Gamma Sampling**
```
γ_i ~ Gamma(α_i, 1)  for i = 1, ..., K
```

**Step 2: Dirichlet Construction**
```
π_i = γ_i / Σⱼ γ_j  for i = 1, ..., K
```

**Step 3: Learning Rate Scaling**
```
lr_i = lr_min + (lr_max - lr_min) × π_i
```

Where:
- `α_i` are the concentration parameters (learnable)
- `K` is the number of parameters
- `lr_min` and `lr_max` define the learning rate bounds

### 2. Adaptive Concentration Updates

The concentration parameters are updated using gradient information through a momentum-based approach inspired by Adam:

**Moment Estimates:**
```
m_t = β₁ × m_{t-1} + (1 - β₁) × ∇_α L
v_t = β₂ × v_{t-1} + (1 - β₂) × (∇_α L)²
```

**Bias Correction:**
```
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
```

**Parameter Update:**
```
α_t = α_{t-1} + η × m̂_t / (√v̂_t + ε)
```

### 3. Gradient Flow

The key innovation is maintaining gradient flow through the sampling process:

```
∂L/∂α = ∂L/∂lr × ∂lr/∂π × ∂π/∂γ × ∂γ/∂α
```

This enables the optimizer to learn which learning rate distributions work best for different parameters.


## Experimental Results
After only 60 epochs on the MNIST dataset using a basic MLP (see dart_utils/models) DART reduced the loss to 0.2776, exhibiting exceptional training stability and demonstrating the potential of probabilistic learning rate adaptation.
## Installation

Clone the repo:
```bash
git clone https://github.com/maticos-dev/dart-optimizer.git
cd dart-optimizer
```

Install Dependencies:
```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install -e .
```

## Usage

### Basic Usage

```python
import torch
import torch.nn as nn
from dartopt import Dart
from dartopt.utils import MLP, Trainer, DartDataBuilder

# Create model and data
model = MLP(input_size=784, output_size=10)
X, y = torch.randn(1000, 784), torch.randint(0, 10, (1000,))
dataset = DartDataBuilder(X, y, device='cpu')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

# Initialize DART optimizer
optimizer = Dart(
    model.parameters(),
    lr=1e-3,           # Base learning rate
    alpha=1.0,         # Dirichlet concentration parameter
    lr_min=1e-6,       # Minimum learning rate
    lr_max=1e-1,       # Maximum learning rate
    betas=(0.9, 0.999) # Adam-style momentum parameters
)

# Training
trainer = Trainer(dataloader, num_epochs=60)
criterion = nn.CrossEntropyLoss()
history = trainer.train(model, optimizer, criterion)
```

### Advanced Configuration

```python
# Custom parameter groups with different settings
optimizer = Dart([
    {'params': model.fc1.parameters(), 'lr': 1e-3, 'alpha': 2.0},
    {'params': model.fc2.parameters(), 'lr': 5e-4, 'alpha': 1.5},
    {'params': model.fc3.parameters(), 'lr': 1e-4, 'alpha': 1.0}
])

# Monitor learning rate samples
lr_samples = optimizer.get_lr_samples()
concentration_params = optimizer.get_concentration_params()
```

### Key Parameters

- `lr`: Base learning rate for scaling the Dirichlet samples
- `alpha`: Initial concentration parameter for the Dirichlet distribution
- `lr_min/lr_max`: Bounds for the sampled learning rates
- `betas`: Momentum parameters for concentration updates (β₁, β₂)
- `eps`: Numerical stability term
- `weight_decay`: L2 regularization coefficient
## Citation
If you use DART in academic work, please cite the following papers, whose insights played an outsize role in the development of this probabilistic optimizer:
- Kingma & Welling, Auto-Encoding Variational Bayes (2014)
- Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts (2017)
- Joo et al., Dirichlet Variational Autoencoder (2019)
## Novel Advantages of DART
While traditional optimizers use deterministic rules, DART introduces informed randomness backed by Bayesian theory. It enhances:
- Exploration via sampling
- Adaptation via gradients through distribution parameters
- Interpretability by representing biases as learnable distributions