import board_recognition as br

import sys
import argparse
import pathlib
import matplotlib.pyplot as mplt
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

    pre_image = br.preprocess(image)

    polygon_image = br.process_image(pre_image)

    fig, (ax1, ax2) = mplt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')

    ax2.imshow(polygon_image, cmap='gray')
    ax2.set_title('Polygon Image')


    file_name = image_file.name
    output_path = pathlib.Path("output") / file_name

    fig.savefig(output_path)
    # mplt.show()

    # cols, rows = polygon_image.shape
    # scale_factor = 64 / max(rows, cols)
    # new_cols = int(cols * scale_factor)
    # new_rows = int(rows * scale_factor)
    # resized = br.downsize_gray(polygon_image, (new_cols, new_rows))

    # fig, (ax1, ax2) = mplt.subplots(1, 2, figsize=(10, 5))
    # ax1.imshow(image)
    # ax1.set_title('Original Image')

    # ax2.imshow(resized, cmap='gray')
    # ax2.set_title('Polygon Image')

    # mplt.show()

    # point_indices = np.argwhere(polygon_image == 1)
    # alpha_shape_points = br.alpha_shape(point_indices, 1.0)
    # print(alpha_shape_points)

