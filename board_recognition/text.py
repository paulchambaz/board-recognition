import board_recognition as br

import matplotlib.pyplot as mplt

import numpy as np
from tqdm import tqdm
import cv2
import json

from shapely.geometry import Polygon
from shapely.geometry import LineString, Point, box

# from shapely.ops import unary_union, cascaded_union
# from skimage.measure import approximate_polygon

def get_text_boxes(image):
    print('computing text boxes')
    pre_image = br.preprocess(image)

    polygon_image = br.process_image(pre_image) / 255

    point_indicies = np.argwhere(polygon_image == 1)
    point_indicies = point_indicies[::7]
    alpha_shape = br.alpha_shape(point_indicies, 0.0)
    alpha_shape_flipped = np.flip(alpha_shape, axis=1)

    quad = compute_quad(alpha_shape_flipped) * image.shape[0] / pre_image.shape[0]
    corrected, matrix = correct_perspective(image, quad)
    pre_corrected = preprocess_masked(corrected)

    smoothed_image = cv2.GaussianBlur(pre_corrected, (5, 5), 0.5)

    gradient_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    horizontal_blur_image = cv2.GaussianBlur(magnitude, (25, 25), 6.0, 0.2)

    stretch_horizontal = stretch(horizontal_blur_image)

    _, binary_image_back = cv2.threshold(stretch_horizontal.astype(np.uint8), 200, 255, cv2.THRESH_BINARY)

    _, binary_image = cv2.threshold(stretch_horizontal.astype(np.uint8), 100, 255, cv2.THRESH_BINARY)
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image)

    stats = stats.tolist()
    stats = [(i,) + tuple(x) for i, x in enumerate(stats)]

    stats = sorted(stats[1:], key=lambda x: x[cv2.CC_STAT_AREA + 1], reverse=True)
    largest_area = stats[len(stats) // 2][cv2.CC_STAT_AREA + 1]

    acceptable_component = []

    for stat in stats:
        label = stat[0]
        area = stat[cv2.CC_STAT_AREA + 1]
        if area >= largest_area * 0.1:
            component_image = np.zeros_like(binary_image, dtype=np.uint8)
            component_image[labels == label] = 255
            acceptable_component.append(component_image)

    print(len(acceptable_component))

    return binary_image

    # for each connected component in the image
    # if it is too small we just discard it (what does too small mean)
    # find its skeleton
    # find its directional vector
    # if the vector is too vertical we just give up
    # if the vector is too small we just give up
    # correct the perspective of the connected component
    # find the bounding box of the connected component
    # if its ratio is too small we just give up
    # correct the perspective of the box back
    # done for loop
    # we should now have all the boxes
    # correct their perspective with the perspective matrix
    # correct their ratio so it fit the original image
    # we are done

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
