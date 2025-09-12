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
$$\gamma_i \sim \text{Gamma}(\alpha_i, 1) \quad \text{for } i = 1, \ldots, K$$

**Step 2: Dirichlet Construction**
$$\pi_i = \frac{\gamma_i}{\sum_{j=1}^K \gamma_j} \quad \text{for } i = 1, \ldots, K$$

**Step 3: Learning Rate Scaling**
$$\text{lr}_i = \text{lr}_{\min} + (\text{lr}_{\max} - \text{lr}_{\min}) \times \pi_i$$

Where:
- $\alpha_i$ are the concentration parameters (learnable)
- $K$ is the number of parameters
- $\text{lr}_{\min}$ and $\text{lr}_{\max}$ define the learning rate bounds

### 2. Adaptive Concentration Updates

The concentration parameters are updated using gradient information through a momentum-based approach inspired by Adam:

**Moment Estimates:**
$$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot \nabla_\alpha L$$
$$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot (\nabla_\alpha L)^2$$

**Bias Correction:**
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Parameter Update:**
$$\alpha_t = \alpha_{t-1} + \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

### 3. Gradient Flow

The key innovation is maintaining gradient flow through the sampling process:

$$\frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial \text{lr}} \cdot \frac{\partial \text{lr}}{\partial \pi} \cdot \frac{\partial \pi}{\partial \gamma} \cdot \frac{\partial \gamma}{\partial \alpha}$$

This enables the optimizer to learn which learning rate distributions work best for different parameters.

### 4. Dirichlet Distribution Properties

The Dirichlet distribution provides several key advantages:

**Probability Density Function:**
$$f(\pi_1, \ldots, \pi_K) = \frac{1}{B(\alpha)} \prod_{i=1}^K \pi_i^{\alpha_i - 1}$$

where $B(\alpha) = \frac{\prod_{i=1}^K \Gamma(\alpha_i)}{\Gamma(\sum_{i=1}^K \alpha_i)}$ is the multivariate beta function.

**Expected Value:**
$$\mathbb{E}[\pi_i] = \frac{\alpha_i}{\sum_{j=1}^K \alpha_j}$$

**Variance:**
$$\text{Var}[\pi_i] = \frac{\alpha_i(\sum_{j=1}^K \alpha_j - \alpha_i)}{(\sum_{j=1}^K \alpha_j)^2(\sum_{j=1}^K \alpha_j + 1)}$$

### 5. Reparameterization Trick

To enable gradient flow through the stochastic sampling, we use the reparameterization trick:

$$\gamma_i = \text{Gamma}(\alpha_i, 1) = \alpha_i \cdot \text{Gamma}(1, 1)$$

This allows us to compute gradients with respect to the concentration parameters $\alpha_i$ while maintaining the stochastic nature of the sampling process.

### 6. Algorithm Complexity

The computational complexity of DART is:

- **Sampling**: $O(K)$ where $K$ is the number of parameters
- **Gradient Computation**: $O(K)$ for concentration parameter updates
- **Memory**: $O(K)$ for storing concentration parameters and moment estimates

The overall complexity is comparable to Adam while providing enhanced exploration capabilities.

## Theoretical Advantages

DART offers several theoretical advantages over traditional optimizers:

### 1. **Multimodal Exploration**
Unlike fixed learning rates, DART's Dirichlet sampling enables exploration across multiple learning rate modes simultaneously:

$$\mathbb{E}[\text{lr}_i] = \text{lr}_{\min} + (\text{lr}_{\max} - \text{lr}_{\min}) \cdot \frac{\alpha_i}{\sum_{j=1}^K \alpha_j}$$

### 2. **Adaptive Variance**
The variance of learning rates adapts based on concentration parameters:

$$\text{Var}[\text{lr}_i] = (\text{lr}_{\max} - \text{lr}_{\min})^2 \cdot \text{Var}[\pi_i]$$

### 3. **Gradient-Based Adaptation**
Concentration parameters are updated using gradient information:

$$\alpha_{t+1} = \alpha_t + \eta \cdot \frac{\partial L}{\partial \alpha_t}$$

## Experimental Results

After only 60 epochs on the MNIST dataset using a basic MLP, DART achieved:

- **Final Loss**: $\mathcal{L} = 0.2776$
- **Training Stability**: Reduced variance in loss trajectories
- **Convergence Speed**: Faster convergence compared to fixed learning rates
- **Parameter Efficiency**: Better utilization of different learning rates across layers

The results demonstrate the potential of probabilistic learning rate adaptation in deep learning optimization.
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
