# Plug-and-Play ADMM for Image Restoration: Fixed-Point Convergence and Applications

## Team 20:  
Kyung Min Ko, Kunal Mamtani, Jonathan Stoschek, Heron Teegarden, Alec Vucsko

### Overview
This project is a Python implementation of the Plug-and-Play (PnP) Alternating Direction Method of Multipliers (ADMM) algorithm, based on the paper "Plug-and-Play ADMM for Image Restoration: Fixed-Point Convergence and Applications" by Chan et al. (2016). The focus of this work is on applying image restoration techniques to improve image quality for autonomous vehicle systems, specifically in the task of identifying traffic signs from noisy, low-resolution images.

### Problem Statement
The paper by Chan et al. proposes improvements to the classic ADMM algorithm, allowing any bounded denoiser to be plugged in while maintaining the algorithm’s stability and convergence. This flexibility makes PnP ADMM suitable for image restoration tasks, where it can effectively deblur or denoise images.

### Project Goals
Our implementation adapts the PnP ADMM algorithm to enhance the visual clarity of traffic sign images for autonomous vehicle systems. The primary goals are:
- Reimplementation of the PnP ADMM in Python.
- Verification of fixed-point convergence.
- Application of the algorithm to improve traffic sign identification in low-resolution, noisy images.

### Applications
The PnP ADMM algorithm can be applied to multiple areas, including:
- **Image Super-Resolution**: Restoring high-quality images from low-resolution data.
- **Single Photon Imaging**: Useful in medical imaging and low-light photography.

### Methodology
We implement the PnP ADMM with improvements that:
1. Include adaptive update rules for robust performance.
2. Use fixed-point convergence to guarantee stability.
3. Apply super-resolution techniques for traffic sign identification.

Our process involves applying Gaussian noise to images, restoring them using PnP ADMM, and testing them with a Faster R-CNN classifier to compare performance against interpolation-based image upscaling.

### Key Results
Our experiments demonstrated:
- Improved image resolution and quality using PnP ADMM.
- A 12% increase in traffic sign classification accuracy when using super-resolution images compared to interpolation-based upscaling.

### References
- Chan, S. H., Wang, X., & Elgendy, O. A. (2016). Plug-and-Play ADMM for image restoration: Fixed-point convergence and applications. *CoRR, abs/1605.01710*.
- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers.

See [ECE50024_Final.pdf](ECE50024_Final.pdf) for more information
