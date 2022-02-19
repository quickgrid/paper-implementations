## Notes

Only tested on a dataset of `2000 cats` and `2000 dogs`.

## Results


## Todo

- Remove duplicated code and simplify training for both teacher, student.
- Do not load other models if not needed to reduce memory.
- Find out why pytorch CIFAR10 dataset fails with loss backwards.
- Add confusion matrix to tensorboard.

## References

- https://keras.io/examples/vision/knowledge_distillation/
- https://arxiv.org/abs/1503.02531
