"""LAVIS BLIP Captioning.

Initial slow version without custom question example. Not all extensions tested if working.

References:
    - https://github.com/salesforce/LAVIS
    - https://github.com/salesforce/LAVIS/blob/main/examples/blip_image_captioning.ipynb
    - https://en.wikipedia.org/wiki/Image_file_format
"""
import os
import argparse
import pathlib

from PIL import Image
from tqdm import tqdm

from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from lavis.models import load_model


def generate_caption(args: argparse.Namespace) -> None:
    args.dest = args.dest or args.src

    if args.ckpt:
        vis_processor = load_processor("blip_image_eval").build(image_size=args.img_size)
        model = load_model(
            name="blip_caption",
            model_type="large_coco",
            is_eval=True,
            device=args.device,
            checkpoint=args.ckpt,
        )
    else:
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip_caption", model_type="large_coco", is_eval=True, device=args.device
        )
        vis_processor = vis_processors['eval']

    ext_check = (".jpg", ".jpeg", ".png")
    if args.ext:
        ext_check += (".jfif", ".webp", ".bmp")

    for root, dirs, files in os.walk(args.src):
        for file in tqdm(files):
            if file.endswith(ext_check):
                img_path = os.path.join(root, file)
                raw_image = Image.open(img_path).convert("RGB")
                image = vis_processor(raw_image).unsqueeze(0).to(args.device)

                if args.beam:
                    caption_list = model.generate(
                        {"image": image},
                        max_length=args.max_len,
                        min_length=args.min_len,
                        top_p=0.9,
                    )
                else:
                    caption_list = model.generate(
                        {"image": image},
                        use_nucleus_sampling=True,
                        num_captions=args.cap,
                        max_length=args.max_len,
                        min_length=args.min_len,
                        top_p=0.9,
                    )

                output = '\n'.join(caption_list) + '\n'
                with open(os.path.join(args.dest, f'{os.path.splitext(file)[0]}.txt'), 'w') as f:
                    f.write(output)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate captions from images.')
    parser.add_argument('-s', '--src', required=True, help="path to source images folder", type=pathlib.Path)
    parser.add_argument('-d', '--dest', help="path to destination caption folder", type=pathlib.Path)
    parser.add_argument('--device', help="use cpu or cuda gpu", default='cuda', type=str)
    parser.add_argument('--ckpt', help="optional custom path to checkpoint", type=pathlib.Path)
    parser.add_argument('--img_size', metavar='--img-size', help="image size for processing", default=384, type=int)
    parser.add_argument('--max_len', metavar='--max-len', help="image size for processing", default=40, type=int)
    parser.add_argument('--min_len', metavar='--min-len', help="image size for processing", default=16, type=int)
    parser.add_argument('--cap', help="number of captions using neucleus sampling", default=3, type=int)
    parser.add_argument('--beam', help="generates single caption without sampling", action='store_true')
    parser.add_argument('--ext', help="extended mode supporting more image types", action='store_true')
    args = parser.parse_args()
    generate_caption(args)


if __name__ == '__main__':
    main()
