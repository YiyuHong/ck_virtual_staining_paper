import os
import sys
from ck_virtual_staining import predict_wsi
import argparse
import time

COPYRIGHT_HOLDER = 'Arontier Inc.'

parser = argparse.ArgumentParser(
    description="""
    Test H&E WSI  
    """,
    epilog='Copyright {} {} All rights reserved.'.format(
        time.strftime("%Y"),
        COPYRIGHT_HOLDER),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('-model_checkpoint', metavar='model-path',
                    type=str, dest='model_path',
                    help='designate a directory model checkpoint path')
parser.add_argument('-input_path', metavar='wsi-input-path',
                    type=str, dest='input_path',
                    help='designate a directory input path')
parser.add_argument('-output_dir', metavar='output-path',
                    type=str, dest='output_path',
                    help='designate a directory output path')
args = parser.parse_args()


# Testing settings
input_file_path = args.input_path
model_file_path = args.model_path
predict_wsi(input_file_path, args.output_path, model_file_path)
