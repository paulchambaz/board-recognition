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

import board_recognition as br

def main():
    board_training_scores = []
    board_training_files = load_files("ground-truth/board/training")
    print('Board training')
    for board in tqdm(board_training_files):
        n = board.stem
        image_file = f"data/{n}.jpg"
        image = load_image(image_file)

        ground_truth = load_polygons_from_json(board)        
        ground_truth = np.array(ground_truth[0]).astype(int)
        image_ground_truth = create_polygon_image(image, ground_truth)

        result = br.get_board_polygon(image)
        image_result = create_polygon_image(image, result)

        iou = calculate_iou(image_ground_truth, image_result)

        # print_fig(n, image, image_ground_truth, image_result, iou, "board")

        board_training_scores.append(iou)
        
    text_training_scores = []
    text_training_files = load_files("ground-truth/text/training")
    print('Text training')
    for text in tqdm(text_training_files):
        n = text.stem
        image_file = f"data/{n}.jpg"
        image = load_image(image_file)
    
        ground_truth = load_polygons_from_json(text)
        if not ground_truth or not ground_truth[0]:
            ground_truth = []
        else:
            ground_truth = [[[int(x), int(y)] for x, y in polygon] for polygon in ground_truth]
    
        image_ground_truth = create_polygons_image(image, ground_truth)
    
        result = br.get_text_boxes(image)
        image_result = create_polygons_image(image, result)
    
        iou = calculate_iou(image_ground_truth, image_result)
    
        # print_fig(n, image, image_ground_truth, image_result, iou, "text")
    
        text_training_scores.append(iou)

    board_test_scores = []
    board_test_files = load_files("ground-truth/board/test")
    print('Board test')
    for board in tqdm(board_test_files):
        n = board.stem
        image_file = f"data/{n}.jpg"
        image = load_image(image_file)

        ground_truth = load_polygons_from_json(board)        
        ground_truth = np.array(ground_truth[0]).astype(int)
        image_ground_truth = create_polygon_image(image, ground_truth)

        result = br.get_board_polygon(image)
        image_result = create_polygon_image(image, result)

        iou = calculate_iou(image_ground_truth, image_result)

        print_fig(n, image, image_ground_truth, image_result, iou, "board")

        board_test_scores.append(iou)

    print_deciles(board_test_scores)

    text_test_scores = []
    text_test_files = load_files("ground-truth/text/test")
    print('Text test')
    for text in tqdm(text_test_files):
        n = text.stem
        image_file = f"data/{n}.jpg"
        image = load_image(image_file)
    
        ground_truth = load_polygons_from_json(text)
        if not ground_truth or not ground_truth[0]:
            ground_truth = []
        else:
            ground_truth = [[[int(x), int(y)] for x, y in polygon] for polygon in ground_truth]
    
        image_ground_truth = create_polygons_image(image, ground_truth)
    
        result = br.get_text_boxes(image)
        image_result = create_polygons_image(image, result)
    
        iou = calculate_iou(image_ground_truth, image_result)
    
        print_fig(n, image, image_ground_truth, image_result, iou, "text")
    
        text_test_scores.append(iou)

        
    print_deciles(text_test_scores)

    training_file = datetime.today().strftime('report/training/report-%Y-%m-%d-%H-%M-%S.txt')
    test_file = datetime.today().strftime('report/test/report-%Y-%m-%d-%H-%M-%S.txt')

    with open(training_file, "w") as file:
        file.write("Summary:\n")
    
        file.write("Board:\n")
        file.write(f"N:{len(board_training_scores)}\n")
        if len(board_training_scores) > 0:
            file.write(f"Mean:{statistics.mean(board_training_scores)}\n")
            file.write(f"Median:{statistics.median(board_training_scores)}\n")
            file.write(f"Worst:{min(board_training_scores)}\n")
            file.write(f"Best:{max(board_training_scores)}\n")
    
        file.write("Text:\n")
        file.write(f"N:{len(text_training_scores)}\n")
        if len(text_training_scores) > 0:
            file.write(f"Mean:{statistics.mean(text_training_scores)}\n")
            file.write(f"Median:{statistics.median(text_training_scores)}\n")
            file.write(f"Worst:{min(text_training_scores)}\n")
            file.write(f"Best:{max(text_training_scores)}\n")

    with open(test_file, "w") as file:
        file.write("Summary:\n")
    
        file.write("Board:\n")
        file.write(f"N:{len(board_test_scores)}\n")
        if len(board_test_scores) > 0:
            file.write(f"Mean:{statistics.mean(board_test_scores)}\n")
            file.write(f"Median:{statistics.median(board_test_scores)}\n")
            file.write(f"Worst:{min(board_test_scores)}\n")
            file.write(f"Best:{max(board_test_scores)}\n")
    
        file.write("Text:\n")
        file.write(f"N:{len(text_test_scores)}\n")
        if len(text_test_scores) > 0:
            file.write(f"Mean:{statistics.mean(text_test_scores)}\n")
            file.write(f"Median:{statistics.median(text_test_scores)}\n")
            file.write(f"Worst:{min(text_test_scores)}\n")
            file.write(f"Best:{max(text_test_scores)}\n")


def load_files(path):
    path = pathlib.Path(path)
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

def create_polygons_image(image, polygon):
    image_pil = Image.fromarray(image)
    size = image_pil.size

    image_polygon = Image.new("L", size)
    draw = ImageDraw.Draw(image_polygon)
    
    for poly in polygon:
        points = [tuple(point) for point in poly]
        draw.polygon(points, fill="white")

    return np.array(image_polygon)

def calculate_iou(image1, image2):
    intersection = np.logical_and(image1, image2)
    union = np.logical_or(image1, image2)
    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    if union_area == 0:
        return 1.0

    iou = intersection_area / union_area
    return iou

def print_fig(n, image, ground_truth, result, iou, location):
    fig, (ax1, ax2, ax3) = mplt.subplots(1, 3, figsize=(10, 5))
    
    ax1.imshow(image)
    ax1.set_title('Original image')
    
    ax2.imshow(ground_truth, cmap='gray')
    ax2.set_title('Ground truth')
    
    ax3.imshow(result, cmap='gray')
    ax3.set_title('Result')
    
    fig.suptitle(f"Image n°{n}: {int(iou * 100)}%")
    
    # mplt.show()
    output_path = f"output/{location}/{n}.jpg"
    fig.savefig(output_path)

    mplt.close(fig)


def print_deciles(scores):
    scores.sort()
    segment_size = len(scores) // 9  # Number of elements in each 10% segment
    
    for i in range(0, len(scores), segment_size):
        segment = scores[i:i+segment_size]  # Get the current segment
        segment_average = sum(segment) / len(segment)  # Calculate the average
        print("Segment", (i // segment_size) + 1, "average:", segment_average)
