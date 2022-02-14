## Notes

Trained on small subset of `MNIST` greyscale images. Also trained on `6000` images of `person, cat, car` with `2000` images each. Tested with `image_size=32, patch_size=8` and `image_size=72, patch_size=6`.

By providing `checkpoint_path` it will resume training from last point. Seems to produce acceptable result within a few epochs without pretraining.

Directory structure for running code should be,

```
dataset_path
  class_1
    img1.jpg
    img2.jpg
    ...
  class_2
    img1.jpg
    img2.jpg
    ...
  class_3
    img1.jpg
    img2.jpg
    ...
  ...
```


## Results

<img src="results/vit.gif" width=50% height=50%>

## Todo

- Add class token to patch embedding.
- Check if implementation is correct.


## References
- https://arxiv.org/abs/2010.11929
- https://keras.io/examples/vision/image_classification_with_vision_transformer/
- https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
- https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
