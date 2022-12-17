"""Tool for viewing npy file.
"""
import argparse
import pathlib

import numpy as np


def view_npy(args: argparse.Namespace) -> None:
    args.dest = args.dest or args.src
    data = np.load(args.src, mmap_mode='r')
    print(data.shape)
    if args.show:
        print(data)
    if args.txt:
        np.savetxt(pathlib.Path(args.dest.parent, f'{args.src.stem}.txt'), data, delimiter=',', fmt='%1.3f')


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate captions from images.')
    parser.add_argument('-s', '--src', required=True, help="path to npy file", type=pathlib.Path)
    parser.add_argument('-d', '--dest', help="path to destination folder", type=pathlib.Path)
    parser.add_argument('--show', help="shows array values in npy file", action='store_true')
    parser.add_argument('--txt', help="output text file with values", action='store_true')
    args = parser.parse_args()
    view_npy(args)


if __name__ == '__main__':
    main()
