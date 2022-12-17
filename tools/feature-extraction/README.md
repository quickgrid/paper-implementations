# Feature Extraction

Any of the methods can be used to generate npy files used for training. [This paper](https://arxiv.org/abs/2211.01324) combines t5, clip text encoder and clip image encoder for optional image.

Chosen libraries in references must installed or cloned and run from correct path to work.

## View Embedding File

### npy_viewer.py

View npy file data, shape and write to file.

> python npy_viewer.py --src "PATH_TO_NPY_FILE"

## Text Features

### t5_embedding.py

Generates embedding in numpy array npy format from text. Requires transformers and downloads chosen model first time to `.cache` folder. More arg options in code.

> python t5_embedding.py --src "PATH_TO_CAPTION_TXT_FOLDER"

## Image and Text Caption Features

### lavis_blip_embedding.py

Uses blip from lavis library to generate embedding from image and its corresponding text caption.

> python lavis_blip_embedding.py --src "PATH_TO_CAPTION_TXT_AND_IMAGE_FOLDER"

### lavis_albef_embedding.py

Uses albef from lavis library to extract features from image and its corresponding text caption.

> python lavis_albef_embedding.py --src "PATH_TO_CAPTION_TXT_AND_IMAGE_FOLDER"

### blip_embedding.py

Uses original BLIP to generate embedding from image and its corresponding text caption.

> python blip_embedding.py --src "PATH_TO_CAPTION_TXT_AND_IMAGE_FOLDER"

### openclip_embedding.py

Uses openclip for feature extraction/embedding from both given image and its corresponding text caption.

> python openclip_embedding.py --src "PATH_TO_CAPTION_TXT_AND_IMAGE_FOLDER"

### openclip_mclip_embedding.py

Uses openclip and mclip for image and its corresponding text caption feature extraction.

> python openclip_mclip_embedding.py --src "PATH_TO_CAPTION_TXT_AND_IMAGE_FOLDER"


## References

- https://github.com/salesforce/LAVIS
- https://github.com/salesforce/BLIP
- https://github.com/mlfoundations/open_clip
- https://github.com/mlfoundations/open_clip/blob/main/tests/test_inference_simple.py
- https://github.com/FreddeFrallan/Multilingual-CLIP
- https://github.com/FreddeFrallan/Multilingual-CLIP/blob/main/Multilingual_CLIP.ipynb
- https://huggingface.co/M-CLIP/XLM-Roberta-Large-Vit-B-16Plus
