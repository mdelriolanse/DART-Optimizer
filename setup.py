from setuptools import setup

setup(
        name="dartopt",
        version="0.1",
        packages=["dartopt", "dartopt.utils", "dartopt.optim"],
        install_requires=["torch", "numpy"],
        author="Mateo del Rio",
        description="Implementation of novel probabilistic learning rate optimizer with learnable dirichlet distribution." 
        )
