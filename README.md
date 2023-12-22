# ODE_unlabel

This repository contains the code for the paper

> [Reconstruction of dynamical systems from data without time labels](https://arxiv.org/abs/2212.04948)

In this paper, we study the method to reconstruct dynamical systems from data without time labels. Data without time labels appear in many applications, such as molecular dynamics,
single-cell RNA sequencing etc. Reconstruction of dynamical system from time sequence data has been studied extensively. However, these methods do not apply if time labels are unknown. Without time labels, sequence data becomes distribution data. Based on this observation, we propose to treat the data as samples from a probability distribution and try to reconstruct the underlying dynamical system by minimizing the distribution loss, sliced Wasserstein distance more specifically. Extensive experiment results demonstrate the effectiveness of the proposed method. 

## Train

To train the network, run

```bash
pytho main.py
```
## Citation


If you use this repository in your research, please consider citing it as:

```bibtex
@article{zeng2023reconstruction,
  title={Reconstruction of dynamical systems from data without time labels},
  author={Zeng, Zhijun and Hu, Pipi and Bao, Chenglong and Zhu, Yi and Shi, Zuoqiang},
  journal={arXiv preprint arXiv:2312.04038},
  year={2023}
}
```