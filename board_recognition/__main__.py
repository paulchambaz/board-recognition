import board_recognition as br

# from gooey import Gooey, GooeyParser
import sys
import argparse
import pathlib
import matplotlib.pyplot as mplt
import matplotlib.image as mimg
from matplotlib.patches import Polygon
import numpy as np

# @Gooey(program_name="Get text from a board")
def main():
    parser = argparse.ArgumentParser(description="Get text from a board")
    # parser = GooeyParser(description="Get text from a board")
    parser.add_argument("-i", "--image", type=str, help="Path to the image")
    # parser.add_argument("-i", "--image", type=str, help="Path to the image", widget="FileChooser")
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

    # polygon = br.get_board_polygon(image)
    #
    # fig, ax = mplt.subplots()
    # ax.imshow(image)
    # polygon_patch = Polygon(polygon, alpha=0.5)
    # ax.add_patch(polygon_patch)
    # x_coords, y_coords = zip(*polygon)
    # ax.scatter(x_coords, y_coords, color='red', marker='x')
    # mplt.show()

    boxes = br.get_text_boxes(image)

    mplt.imshow(boxes, cmap='gray')
    mplt.show()


    # fig, ax = mplt.subplots()
    # ax.imshow(image)
    # polygon_patch = Polygon(boxes, alpha=0.5)
    # ax.add_patch(polygon_patch)
    # x_coords, y_coords = zip(*boxes)
    # ax.scatter(x_coords, y_coords, color='red', marker='x')
    # mplt.show()

