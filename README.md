# Paper-Implementations

Implementation attempts of various AI papers in simple form etc for my learning purposes. These implementations are not meant to be exact and are categorized into a general section. 

These codes should work. Reports for any bugs, mistakes welcome.

**WARNING: Codes may be incomplete, will likely have bugs, mistakes.**

:rocket: Represents I am fairly confident implementation works **(some things may not be same as defined in paper)** on custom dataset and at least part of it is close enough to proposed paper topic. Also mostly tried to follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and [google python style guide](https://google.github.io/styleguide/pyguide.html).

## Pytorch

| Topic | Code |
| --- | --- |
| **Generative Adverserial Networks (GAN)** | :rocket: [Generative Adversarial Networks (GAN)](pytorch/gan) |
|  | :rocket: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)](pytorch/dcgan) |
|  | :rocket: [Wasserstein GAN (WGAN)](pytorch/wgan) |
|  | :rocket: [Improved Training of Wasserstein GANs (WGAN-GP)](pytorch/wgan-gp) |
|  | :rocket: [Conditional Generative Adversarial Nets](pytorch/conditional-wgan) |
|  | [Semantic Image Synthesis with Spatially-Adaptive Normalization (GauGAN/SPADE)](pytorch/gaugan) |
|  | [Progressive Growing of Gans for Improved Quality, Stability, and Variation (ProGAN)](pytorch/progan) |
|  |  |
| **Denoising Diffusion** | :rocket: [Denoising Diffusion Probabilistic Models (DDPM)](pytorch/ddpm) |
|  | :rocket: [Denoising Diffusion Implicit Models (DDIM)](pytorch/ddim) |
|  |  |
| **Multimodal** | [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)](pytorch/imagen) |
|  |  |
| **Transformers, Attention Mechanisms** | :rocket: [Attention is all you need (Transformers, Self Attention, Multi-Head Self Attention)](pytorch/self-attention) |
|  |  |
| **Neural Style Transfer** | [Image Style Transfer Using Convolutional Neural Networks (NST)](pytorch/neural-style-transfer) |
|  |  |
| **Knowledge Distillation** | :rocket: [Distilling the Knowledge in a Neural Network (Knowledge Distillation)](pytorch/knowledge-distillation) |
|  |  |
| **Vision Transformer** | :rocket: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](pytorch/vision_transformer) |
|  |  |
| **Image Segmentation** | [U-Net: Convolutional Networks for Biomedical Image Segmentation (UNet)](pytorch/u-net) |
|  |  |
| **Verification, Recognition, Clutering** <br> TAG: `Siamese Network (Siam)` | :rocket: [FaceNet: A Unified Embedding for Face Recognition and Clustering (Triplet Loss)](pytorch/siamese-triplet-loss) |
|  | :rocket: [Siamese Neural Networks for One-shot Image Recognition](pytorch/siamese-contrastive-loss) <br>  &nbsp; &nbsp;  &nbsp; [Dimensionality Reduction by Learning an Invariant Mapping (Contrastive Loss)](pytorch/siamese-contrastive-loss) |
|  |  |
| **Implicit Representations** | [Implicit Neural Representations with Periodic Activation Functions (SIREN)](pytorch/siren) <br> [COIN: COmpression with Implicit Neural representations](pytorch/siren) |

<!--
## Keras

| Topic | Code |
| --- | --- |
| **Object Detection** | [Focal Loss for Dense Object Detection (RetinaNet)](keras/retinanet) |
|  |  |
-->


## Tools

Various tools useful for custom training. These are not paper implementation.

| Topic | Code |
| --- | --- |
| **Image resize, verification** | :rocket: [Fast full image dataset resize and corrupted, low resolution image remover](tools) |
|  |  |




