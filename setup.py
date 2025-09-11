from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dartopt",
    version="0.1.0",
    author="Mateo del Rio",
    author_email="mateo@example.com",
    description="DART: Dirichlet Adaptive Random Tuning Optimizer for Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maticos-dev/dart-optimizer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    keywords="optimization, deep learning, pytorch, dirichlet, adaptive learning rates",
    project_urls={
        "Bug Reports": "https://github.com/maticos-dev/dart-optimizer/issues",
        "Source": "https://github.com/maticos-dev/dart-optimizer",
        "Documentation": "https://github.com/maticos-dev/dart-optimizer#readme",
    },
)
