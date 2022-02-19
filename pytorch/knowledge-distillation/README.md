## Notes

Only tested on a dataset of `2000 cats` and `2000 dogs`. Dataset must have equal images for accuracy calculation. Distillation performance cannot be compared correctly between runs when separate validation set is not supplied.

There likely will be data leakage from training set to validation when separate validation dataset is not given. A teacher might learn on subset of data that during students training may fall in validaiton set.

Student learns only from teacher when `label_loss_importance = 0.0` otherwise it also learns that amount from labels and rest from soft targets. Student does not seem get very good result on data tested. Maybe student too weak.

### Monitoring Progress

Assuming tensorboard is installed with tensorflow or separately. Monitoring progress in tensorboard in windows,

```
tensorboard --logdir=C:\GAN\logs --host localhost --port 8088
```

In browser head to, http://localhost:8088/ to see tensorboard. Replace, `C:\GAN\logs` with the location of this code where `logs` folder is generated.



## Results

TODO

## Todo

- Test if model is working correctly.
- Remove duplicated code and simplify training for both teacher, student.
- Do not load other models if not needed to reduce memory.
- Find out why pytorch CIFAR10 dataset fails with loss backwards.
- Add confusion matrix to tensorboard.

## References

- https://keras.io/examples/vision/knowledge_distillation/
- https://arxiv.org/abs/1503.02531
