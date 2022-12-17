# Image Caption Generation

Chosen libraries in references must installed or cloned and run from correct path to work.

### lavis_blip_captioning.py

By default generates `N` captions per image. During training for each image randomly 1 of the `N` caption embedding can be picked. More flags are available in code. 

> python lavis_blip_captioning.py --src "PATH_TO_IMAGES"


### blip_captioning.py

Uses original BLIP repo for image captioning.

> python blip_captioning.py --src "PATH_TO_IMAGES"


## References

- https://github.com/salesforce/LAVIS
- https://github.com/salesforce/BLIP
