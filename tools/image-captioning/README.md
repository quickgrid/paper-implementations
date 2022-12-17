# Image Caption Generation

### lavis_blip_captioning.py

By default generates `N` captions per image. During training for each image randomly 1 of the `N` caption embedding can be picked. More flags are available in code. 

> python lavis_blip_captioning.py --src "PATH_TO_IMAGES"


### blip_captioning.py

Uses original BLIP repo for image captioning.

> python blip_captioning.py --src "PATH_TO_IMAGES"
