import board_recognition as br

import matplotlib.pyplot as mplt

import numpy as np
from tqdm import tqdm
import cv2
import json

from shapely.geometry import Polygon
from shapely.geometry import LineString, Point, box
from skimage.morphology import skeletonize

def get_text_boxes(image):
    pre_image = br.preprocess(image)

    polygon_image = br.process_image(pre_image) / 255

    point_indicies = np.argwhere(polygon_image == 1)
    point_indicies = point_indicies[::7]
    alpha_shape = br.alpha_shape(point_indicies, 0.0)
    alpha_shape_flipped = np.flip(alpha_shape, axis=1)

    quad = compute_quad(alpha_shape_flipped) * image.shape[0] / pre_image.shape[0]
    corrected, matrix = correct_perspective(image, quad)
    pre_corrected = preprocess_masked(corrected)

    smoothed_image = cv2.GaussianBlur(pre_corrected, (3, 3), 0.1)

    gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    horizontal_blur_image = cv2.GaussianBlur(magnitude, (25, 25), 6.0, 0.01)

    stretch_horizontal = stretch(horizontal_blur_image)

    _, binary_image = cv2.threshold(stretch_horizontal.astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)

    stats = stats.tolist()
    stats = [(i,) + tuple(x) for i, x in enumerate(stats)]

    stats = sorted(stats[1:], key=lambda x: x[cv2.CC_STAT_AREA + 1], reverse=True)
    largest_area = stats[len(stats) // 2][cv2.CC_STAT_AREA + 1]

    acceptable_components = []

    for stat in stats:
        label = stat[0]
        area = stat[cv2.CC_STAT_AREA + 1]
        if area >= largest_area * 0.2:
            component_image = np.zeros_like(binary_image, dtype=np.uint8)
            component_image[labels == label] = 255
            acceptable_components.append(component_image)

    boxes = []

    for component in acceptable_components:
        skeleton = get_skeleton(component)
        direction = calculate_direction(skeleton)

        if direction > 75 or direction < -75:
            continue

        # TODO: add the rotation
        # rotated = rotate_points(component, direction)

        box = get_bbox(component)
        box = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])

        box = undo_box_perspective(box, matrix)

        boxes.append(box)

    return boxes

def compute_quad(polygon):
    min_x = np.min(polygon[:, 0])
    max_x = np.max(polygon[:, 0])
    min_y = np.min(polygon[:, 1])
    max_y = np.max(polygon[:, 1])

    step = 8

    top_left = min(polygon, key=lambda p: br.distance(p, (min_x, min_y)))
    top_right = min(polygon, key=lambda p: br.distance(p, (max_x, min_y)))
    bottom_right = min(polygon, key=lambda p: br.distance(p, (max_x, max_y)))
    bottom_left = min(polygon, key=lambda p: br.distance(p, (min_x, max_y)))

    top_left = (min_x, top_left[1] - step / 2)
    top_right = (max_x, top_right[1] - step / 2)
    bottom_right = (max_x, bottom_right[1] + step / 2)
    bottom_left = (min_x, bottom_left[1] + step / 2)

    done = False
    quad = None

    while not done:
        len_left = bottom_left[1] - top_left[1]
        len_right = bottom_right[1] - top_right[1]

        if len_left > len_right:
            top_left = (top_left[0], top_left[1] - step)
            bottom_left = (bottom_left[0], bottom_left[1] + step)
        else:
            top_right = (top_right[0], top_right[1] - step)
            bottom_right = (bottom_right[0], bottom_right[1] + step)

        quad = np.array([top_left, top_right, bottom_right, bottom_left])

        if Polygon(polygon).within(Polygon(quad)):
            done = True
    
    done = False
    while not done:
        top_left = (top_left[0], top_left[1] + step)
        quad = np.array([top_left, top_right, bottom_right, bottom_left])
        if not Polygon(polygon).within(Polygon(quad)):
            top_left = (top_left[0], top_left[1] - step)
            quad = np.array([top_left, top_right, bottom_right, bottom_left])
            done = True

    done = False
    while not done:
        top_right = (top_right[0], top_right[1] + step)
        quad = np.array([top_left, top_right, bottom_right, bottom_left])
        if not Polygon(polygon).within(Polygon(quad)):
            top_right = (top_right[0], top_right[1] - step)
            quad = np.array([top_left, top_right, bottom_right, bottom_left])
            done = True

    done = False
    while not done:
        bottom_right = (bottom_right[0], bottom_right[1] - step)
        quad = np.array([top_left, top_right, bottom_right, bottom_left])
        if not Polygon(polygon).within(Polygon(quad)):
            bottom_right = (bottom_right[0], bottom_right[1] + step)
            quad = np.array([top_left, top_right, bottom_right, bottom_left])
            done = True

    done = False
    while not done:
        bottom_left = (bottom_left[0], bottom_left[1] - step)
        quad = np.array([top_left, top_right, bottom_right, bottom_left])
        if not Polygon(polygon).within(Polygon(quad)):
            bottom_left = (bottom_left[0], bottom_left[1] + step)
            quad = np.array([top_left, top_right, bottom_right, bottom_left])
            done = True

    top_left = (top_left[0] + step, top_left[1] + step)
    top_right = (top_right[0] - step, top_right[1] + step)
    bottom_right = (bottom_right[0] - step, bottom_right[1] - step)
    bottom_left = (bottom_left[0] + step, bottom_left[1] - step)

    return np.array([top_left, top_right, bottom_right, bottom_left])

def correct_perspective(image, quad):
    src_pts = quad.reshape(4, 2).astype(np.float32)

    src_pts_ordered = np.zeros_like(src_pts, dtype=np.float32)
    s = src_pts.sum(axis=1)
    src_pts_ordered[0] = src_pts[np.argmin(s)]
    src_pts_ordered[2] = src_pts[np.argmax(s)]
    d = np.diff(src_pts, axis=1)
    src_pts_ordered[1] = src_pts[np.argmin(d)]
    src_pts_ordered[3] = src_pts[np.argmax(d)]

    side_lengths = [
        np.linalg.norm(src_pts_ordered[i] - src_pts_ordered[i - 1])
        for i in range(4)
    ]

    output_height = int(round((side_lengths[0] + side_lengths[2]) / 2))
    output_width = int(round((side_lengths[1] + side_lengths[3]) / 2))

    dst_pts = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height],
        ],
        dtype=np.float32,
    )

    perspective_matrix = cv2.getPerspectiveTransform(src_pts_ordered, dst_pts)

    warped_image = cv2.warpPerspective(image, perspective_matrix, (output_width, output_height))

    return warped_image, perspective_matrix

def preprocess_masked(image):
    cols, rows, _ = image.shape
    max_dim = max(rows, cols)
    if max_dim > 512:
        scale_factor = 512 / max_dim
        new_cols = int(cols * scale_factor)
        new_rows = int(rows * scale_factor)
        resized = cv2.resize(image, (new_rows, new_cols))
    else:
        resized = image

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    return stretch(gray)

def stretch(image):
    nonzero_pixels = image[image > 0]

    lower_percentile = np.percentile(nonzero_pixels, 5)
    upper_percentile = np.percentile(nonzero_pixels, 95)

    stretched_image = np.copy(image)
    stretched_image[image < lower_percentile] = lower_percentile
    stretched_image[image > upper_percentile] = upper_percentile
    stretched_image = cv2.normalize(stretched_image, None, 0, 255, cv2.NORM_MINMAX)

    return stretched_image

def get_skeleton(image):
    return skeletonize(image)

def calculate_direction(image):
    nonzero_pixels = np.nonzero(image)
    x_coords, y_coords = nonzero_pixels

    avg_x = np.mean(x_coords)
    avg_y = np.mean(y_coords)

    centered_x = x_coords - avg_x
    centered_y = y_coords - avg_y

    direction_vector = np.array([np.mean(centered_x), np.mean(centered_y)])
    direction_norm = np.linalg.norm(direction_vector)

    if direction_norm == 0:
        return 90

    direction_vector /= direction_norm

    angle_rad = np.arctan2(direction_vector[1], direction_vector[0])
    angle_deg = np.rad2deg(angle_rad)

    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180

    return angle_deg


def rotate_points(image, angle):
    nonzero_pixels = np.nonzero(image)
    y_coords, x_coords = nonzero_pixels
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[y_coords, x_coords] = 255
    rect = cv2.boundingRect(mask)
    center = (
        rect[0] + rect[2] // 2,
        rect[1] + rect[3] // 2
    )
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

def get_bbox(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return (x, y, x + w, y + h)

def undo_box_perspective(box, perspective_matrix):
    homogenous_box = np.hstack((box, np.ones((box.shape[0], 1), dtype=box.dtype)))
    inverse_perspective_matrix = np.linalg.inv(perspective_matrix)
    unwarped_box = np.dot(inverse_perspective_matrix, homogenous_box.T)
    unwarped_box = (unwarped_box / unwarped_box[2, :]).T[:, :2]
    return unwarped_box
