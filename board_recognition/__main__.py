import board_recognition as br

import sys
import argparse
import pathlib
import matplotlib.image as mimg
import numpy as np

def main():

    parser = argparse.ArgumentParser(description="Get text from a board")
    parser.add_argument("-i", "--image", type=str, help="Path to the image")
    args = parser.parse_args()

    if not args.image:
        parser.print_usage()
        sys.exit(1)

    image_file = pathlib.Path(args.image)
    if not image_file.is_file():
        parser.print_usage()
        sys.exit(1)

    if image_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        parser.print_usage()
        sys.exit(1)


    image = mimg.imread(image_file)

    gray = br.grayscale(image)

    br.get_board_polygon(gray)


