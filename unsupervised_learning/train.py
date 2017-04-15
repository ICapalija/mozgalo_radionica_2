
from lib.convolutional_autoencoder import train

import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train autoencoder.')
    parser.add_argument('--model', dest='pretrained_model',
                        help='initialize with pretrained model', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    train(args.pretrained_model)