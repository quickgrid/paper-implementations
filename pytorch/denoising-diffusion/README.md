# Denoising Diffusion

Annotated implementation of DDPM (Denoising Diffusion Probabilistic Model). Only slow sampling is implemented so far with both train and test timesteps equal to `T`. 

It may require atleast half or an hour to generate something recognizable. Only `64 x 64` resolution is tested. 

### Features

- Annotated code with example and paper reference.
- Test code for model architecture, expected output.
- Tried to follow PEP-8 style code.
- Save and load model, ema model.
- Sampling with pretrained weights.
- Stochastic weight averaging (SWA) example and implementation. (Not tested)
- EMA (Exponential Moving Average) model.
- Linear, cosine noise scheduling.
- Learning rate scheduling.
- Sinusoidal positional embedding with dropout.
- Precalculated values for faster sampling.
- Mixed precision training.
- Gradient accumulation for large minibatch training.
- UNet with Attention layers.

<br>

## Results


<br>

## Pretrained Checkpoints

<br>

## Todo

- Try to implement ddim.
- Class conditional generation.
- Classifier Free Guidance (CFG).
- Save EMA step number with checkpoint.
- Add super resolution with unet like imagen for 4X upsampling, `64x64 => 256x256 => 1024x1024`.

<br>


## References
- DDPM paper, https://arxiv.org/pdf/2006.11239.pdf.
- DDIM paper, https://arxiv.org/pdf/2010.02502.pdf.
- Annotated Diffusion, https://huggingface.co/blog/annotated-diffusion.
- Keras DDIM, https://keras.io/examples/generative/ddim/.
- Implementation, https://www.youtube.com/watch?v=TBCRlnwJtZU.
- Implementation, https://github.com/dome272/Diffusion-Models-pytorch.
- Postional embedding, http://nlp.seas.harvard.edu/annotated-transformer/.
- Attention paper, https://arxiv.org/pdf/1706.03762.pdf.
- Transformers, https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
