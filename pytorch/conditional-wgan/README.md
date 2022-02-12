## Notes

Pytorch implementation of `conditional gan` with `wgan-gp`. Tested so far only on mnist. Trained with a very small subset of mnist. Results are decent on mnist after `30 epochs`. 

Training can be monitored on tensorboard. Instruction can be found on `wgan-gp` implementation readme.

## Results

![MNIST images](results/conditional_wgan_mnist.gif "Generated Fake MNIST Images")

![Generated Fake MNIST Images](results/conditional_generated.png "Generated Fake MNIST Images")
![Real MNIST Images](results/ground_truth.png "MNIST real Images")


## Todo

- [ ] Test to see if works on other datasets.

## References

- Conditional Generative Adversarial Nets, https://arxiv.org/abs/1411.1784
