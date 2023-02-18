"""Generate t5 embedding from text.

Initial slow version.
"""
import argparse
import glob
import os.path
import pathlib

from tqdm import tqdm
import numpy as np

from t5 import t5_encode_text, MAX_LENGTH


def generate_embedding(args: argparse.Namespace) -> None:
    args.dest = args.dest or args.src
    max_len = args.max_len or MAX_LENGTH

    images_base_path = args.src
    file_list = glob.glob(os.path.join(images_base_path, '*.txt'))
    print(len(file_list))

    embedding_output_path = pathlib.Path(args.dest, 'embedding')
    mask_output_path = pathlib.Path(args.dest, 'mask')
    embedding_output_path.mkdir(parents=True, exist_ok=True)
    mask_output_path.mkdir(parents=True, exist_ok=True)

    for _, text_file_path in enumerate(tqdm(file_list)):
        with open(text_file_path) as f:
            lines = f.readlines()

            text_file_path, file_name = os.path.split(text_file_path)
            file_name, file_ext = os.path.splitext(file_name)

            out = t5_encode_text(
                lines, name=f'{args.type}-{args.arch}', return_attn_mask=args.attn_mask
            )

            if args.attn_mask:
                embedding, mask = out
            else:
                embedding = out

            embedding = embedding.cpu().numpy()
            # if not args.no_pad:
            #     embedding = np.pad(
            #         embedding,
            #         pad_width=((0, 0), (0, max_len - embedding.shape[1]), (0, 0)),
            #         mode='constant',
            #         constant_values=0
            #     )

            np.save(os.path.join(embedding_output_path, f'embedding_{file_name}.npy'), embedding)
            if args.attn_mask:
                mask = mask.cpu().numpy()
                # if not args.no_pad:
                #     mask = np.pad(
                #         mask,
                #         pad_width=((0, 0), (0, max_len - mask.shape[1])),
                #         mode='constant',
                #         constant_values=0
                #     )
                np.save(os.path.join(mask_output_path, f'mask_{file_name}.npy'), mask)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate embeddings from text.')
    parser.add_argument('-s', '--src', required=True, help="path to source caption folder", type=pathlib.Path)
    parser.add_argument('-d', '--dest', help="path to destination embedding folder", type=pathlib.Path)
    parser.add_argument('--device', help="use cpu or cuda gpu", default='cuda', type=str)
    parser.add_argument('--max-len', help="max tokenization length", default=96, type=int)
    parser.add_argument('--attn-mask', help="generate attention mask npy", action='store_true')
    # parser.add_argument('--no-pad', help="disables pad with 0 to match max-len size", action='store_true')
    parser.add_argument(
        '--type', help='type t5 or t5 1.1', default='google/t5-v1_1', choices=['google/t5-v1_1', 't5']
    )
    parser.add_argument(
        '--arch', help='t5 architecture choices (not all)', default='base', choices=['small', 'base', 'large']
    )
    args = parser.parse_args()
    generate_embedding(args)


if __name__ == '__main__':
    main()
