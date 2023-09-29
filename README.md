# This repository
This repository contains the code for the paper

Fast SE(n) Equivariance in Position Orientation Space

# Conda environment
In order to run the code in this repository install the following conda environment
Conda env
```
conda create --yes --name ponita python=3.10 numpy scipy matplotlib
conda activate ponita
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pyg==2.2.0 -c pyg -y
pip3 install wandb
pip3 install pytorch_lightning==1.8.6
```

# Main files

# The Siva Library
# :hand: siva :trident:
**S**E(d) **I**nvariant **V**ector **A**ttributes (:hand: **SIVA** :trident:) of tuples of points in the homogeous spaces $M=\mathbb{R}^d$, $M=\mathbb{R}^d \times S^{d-1}$, or $M=SE(d)$. We consider the cases $d=2$ and $d=3$. The library thus allows for the computation of invariant attributes between tuples of points in
* **Position space** $\mathbb{R}^2$ and $\mathbb{R}^3$ ,
* **Position-Orientation space** $\mathbb{R}^2 \times S^1 \equiv SE(2)$ and $\mathbb{R}^3 \times S^2$ ,
* **Groups** $SE(2)$ and $SE(3)$ .
