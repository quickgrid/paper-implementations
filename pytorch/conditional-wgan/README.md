## Notes

Pytorch implementation of `conditional gan` with `wgan-gp`. Tested so far only on mnist. Trained with a very small subset of mnist. Results are decent on mnist after `30 epochs`. 

Conditional generation can be performed easily with integers representing `folder/class`. No one hot encoding needed, just giving integer will generate sample from that class. Modify following to generate samples from that class. 

```
noise = torch.randn((ncols, z_dim, 1, 1), device=device)
labels = torch.LongTensor([2, 9, 7, 1, 5, 3]).to(device)
```

Number of labels must match `ncols` which is the mini batch size. For mnist the labels are 0 to 9. For cats and dogs the labels will be `0, 1` so on.

Training can be monitored on tensorboard. Instruction can be found on `wgan-gp` implementation readme.

## Results

<img src="results/conditional_wgan_mnist.gif" width=45% height=45%>

### Conditional Generation with User Input

<img src="results/conditional generation.png" width=90% height=90%>

### Generated and Real

<img src="results/conditional_generated.png" width=45% height=50%> <img src="results/ground_truth.png" width=45% height=50%>

## Todo

- Test to see if works on other datasets.
- Add checkpoint saving and reload code from wgap-gp implementation.

## References

- Conditional Generative Adversarial Nets, https://arxiv.org/abs/1411.1784
