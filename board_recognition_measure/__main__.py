"""This file is used to measure the performance of our board detection
algorithms"""

import matplotlib.image as mimg
import matplotlib.pyplot as mplt

import pathlib
import statistics
import random
import json
from datetime import datetime
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np

def main():
    files = load_files("data")
    board_scores = []
    text_scores = []
    schema_scores = []
    
    print("Board")
    for file in tqdm(files):
        n = file.stem
        image = load_image(str(file))

        board = f"ground-truth/board/{n}.json"
        board_poly = load_polygons_from_json(board)        
        board_poly = np.array(board_poly[0]).astype(int)
        image_labelme = create_polygon_image(image, board_poly)

        mplt.imshow(image_labelme, cmap='gray')
        mplt.title(f"image {n}")
        mplt.axis('off')
        mplt.show()

        if not board_poly.size == 0:
            continue

        board_scores.append(random.random())

    print("Text")
    for file in tqdm(files):
        n = file.stem
        image = load_image(str(file))

        text = f"ground-truth/text/{n}.json"
        text_poly = load_polygons_from_json(text)
        if not text_poly:
            continue

        text_scores.append(random.random())

    print("Schema")
    for file in tqdm(files):
        n = file.stem
        image = load_image(str(file))

        schema = f"ground-truth/schema/{n}.json"
        schema_poly = load_polygons_from_json(schema)
        if not schema_poly:
            continue

        schema_scores.append(random.random())
    
    summary_file = datetime.today().strftime('report/report-%Y-%m-%d-%H-%M-%S.txt')


    with open(summary_file, "w") as file:
        file.write("Summary:\n")

        file.write("Board:\n")
        file.write(f"N:{len(board_scores)}\n")
        if len(board_scores) > 0:
            file.write(f"Mean:{statistics.mean(board_scores)}\n")
            file.write(f"Median:{statistics.median(board_scores)}\n")
            file.write(f"Worst:{min(board_scores)}\n")
            file.write(f"Best:{max(board_scores)}\n")

        file.write("Test:\n")
        file.write(f"N:{len(text_scores)}\n")
        if len(text_scores) > 0:
            file.write(f"Mean:{statistics.mean(text_scores)}\n")
            file.write(f"Median:{statistics.median(text_scores)}\n")
            file.write(f"Worst:{min(text_scores)}\n")
            file.write(f"Best:{max(text_scores)}\n")

        file.write("Schema:\n")
        file.write(f"N:{len(schema_scores)}\n")
        if len(schema_scores) > 0:
            file.write(f"Mean:{statistics.mean(schema_scores)}\n")
            file.write(f"Median:{statistics.median(schema_scores)}\n")
            file.write(f"Worst:{min(schema_scores)}\n")
            file.write(f"Best:{max(schema_scores)}\n")


def load_files(path):
    path = pathlib.Path("data")
    return [file for file in path.iterdir() if file.is_file()]


def load_image(path):
    return  mimg.imread(path)


def load_polygons_from_json(json_file_path):
    polygons = []
    try:
        f = open(json_file_path, 'r')
        json_data = json.load(f)
    except OSError:
        return polygons
    for shape in json_data['shapes']:
        polygons.append(shape['points'])
    return polygons


def is_point_inside_polygon(point, polygon):
    """ Checks if a point is inside a polygon using the ray casting algorithm
    @param point A tuple (x, y) representing the point to check
    @param polygon A list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)]
    representing the vertices of the polygon
    @returns True if the point is inside the polygon, False otherwise
    """
    num_intersections = 0
    x, y = point
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if y1 == y2:  # horizontal edge, skip
            continue
        if y1 > y2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        if y < y1 or y >= y2:  # outside y-range of edge, skip
            continue
        # compute x-coordinate of intersection
        x_intercept = (y - y1) * (x2 - x1) / (y2 - y1) + x1
        if x_intercept > x:
            num_intersections += 1
    return num_intersections % 2 == 1


def get_polygon_image(polygon, image):
    """Returns a binary image of the size of the image with the polygon highlighted
    @image The image on which to draw the polygon
    @param polygon A list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)]
    representing the vertices of the polygon
    @param image The image (only used for its shape)
    """
    polygon_image = np.zeros_like(image)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if is_point_inside_polygon((i, j), polygon):
                polygon_image[j, i] = 255
    return polygon_image


def create_polygon_image(image, board_polygon):
    """Return a binary image of the polygon fromn the labeled polygon
    """
    image_pil = Image.fromarray(image)
    size = image_pil.size

    image_polygon = Image.new("L", size)
    draw = ImageDraw.Draw(image_polygon)
    board_polygon = list(map(tuple, board_polygon))
    draw.polygon(board_polygon, fill="white")

    return np.array(image_polygon)
