## Denoising Diffusion

Annotated implementation of DDPM (Denoising Diffusion Probabilistic Model).

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

<br>

## Results


<br>


## Todo

- Try to implement ddim.
- Class conditional generation.
- Classifier Free Guidance (CFG).
- Save EMA step number with checkpoint.

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
