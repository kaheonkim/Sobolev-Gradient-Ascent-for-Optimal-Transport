# Sobolev Gradient Ascent for Optimal Transport: Barycenter Optimization and Convergence Analysis
link : https://arxiv.org/abs/2505.13660

**Kaheon Kim, Bohan Zhou, Changbo Zhu, and Xiaohui Chen**  
*International Conference on Learning Representation (ICLR), 2026*

---

This repository contains the code and experiments for our ICLR 2026 paper, "Sobolev Gradient Ascent for Optimal Transport: Barycenter Optimization and Convergence Analysis". 

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{2505.13660,
  title={Sobolev Gradient Ascent for Optimal Transport: Barycenter Optimization and Convergence Analysis},
  author={Kaheon Kim and Bohan Zhou and Changbo Zhu and Xiaohui Chen},
  booktitle={International Conference on Learning Representation (ICLR)},
  year={2026}
}
```

## Abstract

This paper introduces a new constraint-free concave dual formulation for the Wasserstein barycenter. Tailoring the vanilla dual gradient ascent algorithm to the Sobolev geometry, we derive a scalable Sobolev gradient ascent (SGA) algorithm to compute the barycenter for input distributions supported on a regular grid. Despite the algorithmic simplicity, we provide a global convergence analysis that achieves the same rate as the classical subgradient descent methods for minimizing nonsmooth convex functions in the Euclidean space. A central feature of our SGA algorithm is that the computationally expensive c-concavity projection operator enforced on the Kantorovich dual potentials is unnecessary to guarantee convergence, leading to significant algorithmic and theoretical simplifications over all existing primal and dual methods for computing the exact barycenter. Our numerical experiments demonstrate the superior empirical performance of SGA over the existing optimal transport barycenter solvers.

## Dependencies

- numpy
- matplotlib
- scipy
- BFM
- POT(for comparison)
- kagglehub
- Pillow
- scikit-image
- pybind11
