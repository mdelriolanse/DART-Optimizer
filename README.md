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

1. **Learning Rate Sampling**
For each parameter tensor, we first sample candidate learning rates from a normal distribution:

![Logo](https://latex.codecogs.com/svg.image?{lrcandidates}\sim\mathcal{N}(\mu,\sigma^2))

These are then clamped to a valid range:


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