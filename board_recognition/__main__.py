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

    polygon = br.get_board_polygon(image)
    
    fig, ax = mplt.subplots()
    ax.imshow(image)
    polygon_patch = Polygon(polygon, alpha=0.5)
    ax.add_patch(polygon_patch)
    x_coords, y_coords = zip(*polygon)
    ax.scatter(x_coords, y_coords, color='red', marker='x')
    mplt.show()

    # poly_points = br.load_polygons_from_json('ground-truth/board/10.json')
    #
    # pre_image = br.preprocess(image)
    #
    # polygon_image = br.process_image(pre_image) / 255
    #
    # mplt.imshow(polygon_image, cmap='gray')
    # mplt.show()
    #
    # point_indices = np.argwhere(polygon_image == 1)
    # point_indices = point_indices[::31]
    #
    # shape_image = br.create_point_image(pre_image, point_indices)
    #
    # mplt.imshow(shape_image)
    # mplt.show()
    #
    # alpha_shape_points = br.alpha_shape(point_indices, 0.0)

    # shape_image = br.create_point_image(pre_image, alpha_shape_points)

    # mplt.imshow(shape_image)
    # mplt.show()

    exit(0)

    # file_name = image_file.name
    # output_path = pathlib.Path("output") / file_name


    alpha_shape_points = br.alpha_shape(point_indices, 0.0)

    shape_image = br.create_point_image(pre_image, alpha_shape_points)

    mplt.imshow(shape_image, cmap='gray')
    mplt.show()

    shape_image = br.create_polygon_image(pre_image, alpha_shape_points)
    
    mplt.imshow(shape_image, cmap='gray')
    mplt.show()

    # fig, (ax1, ax2) = mplt.subplots(1, 2, figsize=(10, 5))
    # ax1.imshow(image)
    # ax1.set_title('Original Image')
    #
    # ax2.imshow(resized, cmap='gray')
    # ax2.set_title('Polygon Image')
    #
    # mplt.show()
