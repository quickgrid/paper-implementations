## Notes

Tested on [Facade dataset](https://cmp.felk.cvut.cz/~tylecr1/facade/).

## Results


## Todo

- ~~Fix KL loss becomes nan.~~
- Fix training not converging.
- ~~Fix after some iterations combined generator, vgg, feature loss becomes nan.~~
- ~~Discriminator backward fails due to nan.~~
- Check if architecture properly matches code.
- Pass device type through function.
- Modify to try to generate and match mask also as loss.

## References

- https://arxiv.org/abs/1903.07291
- https://keras.io/examples/generative/gaugan/
- https://github.com/quickgrid/AI-Resources/blob/master/resources/ai-notes/gaugan-series.md
