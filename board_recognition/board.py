import cv2
import numpy as np
from alphashape import alphashape
from shapely.geometry import Polygon


def get_board(image, concave=True):
    scale, resize = resize_image(image, 512)
    preprocess = preprocess_image(resize)
    gaussian = gaussian_smoothing(preprocess, 5, 11)
    gradient = gradients(gaussian)
    gaussian = gaussian_smoothing(gradient, 5, 9)

    _, binary_image = cv2.threshold(
        gradient.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    binary_image = fill_holes(binary_image, size=9)

    inverted = cv2.bitwise_not(binary_image)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=4)

    components = filter_components(inverted, labels, stats, area_ratio_threshold=0.5)
    component = merge_components(inverted, components)

    filled = fill_holes(component, size=5)
    inverted = 255 - filled

    opened = fill_holes(inverted, size=5)
    filled = 255 - opened

    filled = flood_fill_ext(filled)

    edges = cv2.Canny(filled.astype(np.uint8), threshold1=30, threshold2=100)
    edges = edges[1:-1, 1:-1]
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    point_indicies = np.argwhere(edges == 255)
    point_indicies = point_indicies[::59]
    polygon = alpha_shape(edges, point_indicies, solver=concave)

    polygon = [[int(p[0] / scale), int(p[1] / scale)] for p in polygon]
    board_polygon = np.array(polygon, dtype=np.int32)

    return (board_polygon, create_polygon_image(image, board_polygon))


def resize_image(image, max_size):
    cols, rows, _ = image.shape
    max_dim = max(rows, cols)

    if max_dim <= max_size:
        return (1, image)

    scale_factor = max_size / max_dim
    new_cols = int(cols * scale_factor)
    new_rows = int(rows * scale_factor)
    return (scale_factor, cv2.resize(image, (new_rows, new_cols)))


def restore_size_image(image, scale):
    cols, rows = image.shape[:2]

    new_cols = int(cols / scale)
    new_rows = int(rows / scale)

    return cv2.resize(image, (new_rows, new_cols))


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    min_val = np.min(gray)
    max_val = np.max(gray)
    contrasted = cv2.normalize(gray, None, min_val, max_val, cv2.NORM_MINMAX)

    return contrasted


def gaussian_smoothing(image, kernel_size=3, sigma=1.0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def gradients(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    return magnitude


def fill_holes(image, size=5):
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, morph_kernel)
    return closed_image


def filter_components(inverted, labels, stats, area_ratio_threshold):
    stats = stats.tolist()
    stats = [(i,) + tuple(x) for i, x in enumerate(stats)]
    stats = sorted(stats[1:], key=lambda x: x[cv2.CC_STAT_AREA + 1], reverse=True)

    largest_area = stats[0][cv2.CC_STAT_AREA + 1]

    huge_components = []
    for i, stat in enumerate(stats):
        if stat[cv2.CC_STAT_AREA + 1] >= largest_area * area_ratio_threshold:
            component_image = np.zeros_like(inverted, dtype=np.uint8)
            component_image[labels == stat[0]] = 255
            huge_components.append(component_image)

    return huge_components


def merge_components(image, components):
    if len(components) == 1:
        return components[0]

    if len(components) == 2:
        # contained components
        bounding_boxes = [get_bounding_box(component) for component in components]
        contained_components = []
        for i in range(len(bounding_boxes)):
            for j in range(len(bounding_boxes)):
                if i != j and is_inside(
                    bounding_boxes[i], bounding_boxes[j], margin=10
                ):
                    contained_components.append(components[i])
                    break

        if len(contained_components) == 1:
            return contained_components[0]

        # side by side components
        centroids = [get_centroid(component) for component in components]
        is_about_same_height = abs(centroids[0][1] - centroids[1][1]) < image.shape[0] * .1
        is_left_and_right = centroids[0][0] < image.shape[1] * .5 and centroids[1][0] > image.shape[1] * .5
        is_right_and_left = centroids[1][0] < image.shape[1] * .5 and centroids[0][0] > image.shape[1] * .5

        if is_about_same_height and (is_left_and_right or is_right_and_left):
            merged_image = np.zeros_like(components[0], dtype=np.uint8)
            for component in components:
                merged_image = np.bitwise_or(merged_image, component)
            return merged_image


    target_x = image.shape[1] * 0.5
    target_y = image.shape[0] * 0.6

    return min(
        components,
        key=lambda c: distance(get_centroid(c), (target_x, target_y)),
    )


def get_bounding_box(image):
    non_zero_indices = np.nonzero(image)
    x_min, x_max = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    y_min, y_max = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])

    return x_min, x_max, y_min, y_max


def is_inside(inner_box, outer_box, margin=0):
    return (
        inner_box[0] + margin >= outer_box[0]
        and inner_box[1] - margin <= outer_box[1]
        and inner_box[2] + margin >= outer_box[2]
        and inner_box[3] - margin <= outer_box[3]
    )


def get_centroid(image):
    moments = cv2.moments(image)
    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]

    return centroid_x, centroid_y


def distance(p1, p2):
    return (
        (float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2
    ) ** 0.5


def flood_fill_ext(image):
    border_size = 1
    padded_image = np.pad(
        image, pad_width=border_size, mode="constant", constant_values=0
    )

    h, w = padded_image.shape[:2]
    temp_image = np.zeros((h + 2, w + 2), dtype=np.uint8)

    flood_filled = padded_image.copy()

    cv2.floodFill(
        flood_filled,
        temp_image,
        (0, 0),
        newVal=255,
    )
    invert_flood = 255 - flood_filled
    image = padded_image + invert_flood
    return image


def alpha_shape(image, points, solver=True):
    if solver:
        alpha = 0.00
        while True:
            try:
                alpha_polygon = alphashape(points, alpha=alpha)
                exterior_coords = np.array(alpha_polygon.exterior.coords)
                exterior_coords = exterior_coords[:, [1, 0]]
                alpha += 0.001
            except Exception:
                alpha -= 0.001
                break

        if not alpha == 0.0:
            alpha_polygon = alphashape(points, alpha=alpha * 0.05)
            exterior_coords = np.array(alpha_polygon.exterior.coords)
            exterior_coords = exterior_coords[:, [1, 0]]
    else:
        alpha_polygon = alphashape(points, alpha=0)
        exterior_coords = np.array(alpha_polygon.exterior.coords)
        exterior_coords = exterior_coords[:, [1, 0]]

    return exterior_coords


def create_polygon_image(image, board_polygon):
    height, width = image.shape[:2]
    image_polygon = np.zeros((height, width), dtype=np.uint8)

    board_polygon = board_polygon.reshape((-1, 1, 2)).astype(np.int32)

    cv2.fillPoly(image_polygon, [board_polygon], color=(255))
    return image_polygon
