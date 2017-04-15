
from lib.convolutional_autoencoder import extract

import argparse
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test autoencoder.')
    parser.add_argument('--model', dest='pretrained_model',
                        help='initialize with pretrained model',
                        default=None, type=str)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    extract(args.pretrained_model, args.vis)