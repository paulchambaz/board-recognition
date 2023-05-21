import board_recognition as br

from gooey import Gooey, GooeyParser
import sys
import argparse
import pathlib
import matplotlib.pyplot as mplt
import matplotlib.image as mimg
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

    pre_image = br.preprocess(image)

    polygon_image = br.process_image(pre_image)

    # fig, (ax1, ax2) = mplt.subplots(1, 2, figsize=(10, 5))
    # ax1.imshow(image)
    # ax1.set_title('Original Image')
    #
    # ax2.imshow(polygon_image, cmap='gray')
    # ax2.set_title('Polygon Image')

    file_name = image_file.name
    output_path = pathlib.Path("output") / file_name

    point_indices = np.argwhere(polygon_image == 1)
    point_indices = point_indices[::31]

    alpha_shape_points = br.alpha_shape(point_indices, 10)

    shape_image = br.create_point_image(pre_image, alpha_shape_points)

    mplt.imshow(shape_image)
    mplt.show()

    # fig, (ax1, ax2) = mplt.subplots(1, 2, figsize=(10, 5))
    # ax1.imshow(image)
    # ax1.set_title('Original Image')
    #
    # ax2.imshow(resized, cmap='gray')
    # ax2.set_title('Polygon Image')
    #
    # mplt.show()
