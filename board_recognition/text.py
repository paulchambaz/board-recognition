import matplotlib.pyplot as mplt
import numpy as np
from shapely.geometry import Polygon
import cv2

from .board import get_board, resize_image
from .board import gaussian_smoothing, gradients, fill_holes

def get_text(image):
    poly_board, board = get_board(image, concave=False)

    scale, resize = resize_image(image, 512)

    poly_board = [ [p[0] * scale, p[1] * scale] for p in poly_board ]
    poly_board = np.array(poly_board, dtype=np.int32)
 
    quad = compute_quad(poly_board) / scale

    corrected_matrix, corrected_width, corrected_height = correct_perspective(image, quad)
    default_matrix, default_width, default_height = default_perspective(image, quad)

    t = -1

    mixed_matrix = (t * corrected_matrix + (1 - t) * default_matrix)
    mixed_width = int(t * corrected_width + (1 - t) * default_width)
    mixed_height = int(t * corrected_height + (1 - t) * default_height)

    default_image = cv2.warpPerspective(image, default_matrix, (default_width, default_height))
    mixed_image = cv2.warpPerspective(image, mixed_matrix, (mixed_width, mixed_height))
    corrected_image = cv2.warpPerspective(image, corrected_matrix, (corrected_width, corrected_height))

    default_mask = cv2.warpPerspective(board, default_matrix, (default_width, default_height))
    mixed_mask = cv2.warpPerspective(board, mixed_matrix, (mixed_width, mixed_height))
    corrected_mask = cv2.warpPerspective(board, corrected_matrix, (corrected_width, corrected_height))

    default_image = process(default_image, default_mask)
    mixed_image = process(mixed_image, mixed_mask)
    corrected_image = process(corrected_image, corrected_mask)

    fig, axs = mplt.subplots(2, 3, figsize=(10, 5)) 

    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 1].imshow(board, cmap='gray')
    axs[0, 2].imshow(image, cmap='gray')

    quad_polygon = mplt.Polygon(quad, color='red', alpha=0.2)
    axs[0, 2].add_patch(quad_polygon)

    for point in quad:
        axs[0, 2].plot(*point, marker='o', color='red')

    axs[1, 1].imshow(default_image, cmap='gray')
    axs[1, 0].imshow(mixed_image, cmap='gray')
    axs[1, 2].imshow(corrected_image, cmap='gray')

    axs[0, 0].axis('off')
    axs[0, 1].axis('off')
    axs[0, 2].axis('off')
    axs[1, 0].axis('off')
    axs[1, 1].axis('off')
    axs[1, 2].axis('off')

    mplt.show()

    return board


def process(image, mask):
    image = blacken_mask(image, mask)

    hue, saturation, value = preprocess_image(image)

    hue = get_gradients(hue, mask)
    saturation = get_gradients(saturation, mask)
    value = get_gradients(value, mask)

    result = (hue / 3 + saturation / 3 + value / 3)

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = np.uint8(result)

    result = fill_holes(result, size=5)

    _, binary = cv2.threshold(
        result.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    binary = fill_holes(binary, size=5)
    inverted = cv2.bitwise_not(binary)
    inverted = fill_holes(inverted, size=5)
    binary = cv2.bitwise_not(inverted)

    n = 2
    rlsa = binary

    kernel = np.array([
        [ 0, 0, 0, 0, 0 ],
        [ 0, 0, 0, 0, 0 ],
        [ 0, 1, 1, 1, 0 ],
        [ 0, 0, 0, 0, 0 ],
        [ 0, 0, 0, 0, 0 ],
    ], np.uint8)

    for i in range(n):
        rlsa = apply_rlsa(rlsa, 25)
        rlsa = defill_holes(rlsa, 5)
        rlsa = cv2.dilate(rlsa, kernel, iterations=1)
        inverted = cv2.bitwise_not(rlsa)
        inverted = apply_rlsa(inverted, 25)
        rlsa = cv2.bitwise_not(inverted)


    kernel = np.array([
        [ 0, 0, 1, 0, 0 ],
        [ 1, 1, 1, 1, 1 ],
        [ 1, 1, 1, 1, 1 ],
        [ 1, 1, 1, 1, 1 ],
        [ 0, 0, 1, 0, 0 ],
    ], np.uint8)

    rlsa = cv2.dilate(rlsa, kernel, iterations=1)

    inverted = cv2.bitwise_not(rlsa)
    inverted = fill_holes(inverted, size=5)
    rlsa = cv2.bitwise_not(inverted)

    return rlsa

def apply_rlsa(image, threshold):
    cols, rows = image.shape
    smeared = np.copy(image)

    for y in range(cols):
        for x in range(rows):
            if image[y, x] == 255:
                count = threshold
                for i in range(count - 1, 0, -1):
                    if x + i >= rows:
                        continue
                    if image[y, x + i] == 255:
                        for j in range(1, i):
                            smeared[y, x + j] = 255
                        break
                    else:
                        continue

                    if count <= 0:
                        break

    return smeared

    # for i, row in enumerate(image):
    #     zero_count = 0
    #     start_index = 0
    #     for j, val in enumerate(row):
    #         if val != 0:
    #             print(val)
    #         # if val == 0:
    #         #     if zero_count == 0:
    #         #         start_index = j
    #         #     zero_count += 1
    #         # else:
    #         #     if 0 < zero_count <= threshold:
    #         #         smeared[i, start_index:j] = 0
    #             # zero_count = 0
    #     return smeared

def get_gradients(image, mask):
    scale, resize = resize_img(image, 512)
    cols, rows = image.shape
    mask = cv2.resize(mask, (int(rows * scale), int(cols * scale)))
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8), iterations=1)

    gaussian = gaussian_smoothing(resize, 3, 1.2)
    gradient = gradients(gaussian)

    mask_bool = mask.astype(bool)
    gradient = gradient * mask

    return gradient

def preprocess_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    min_hue, max_hue = get_low_high(hue)
    hue = cv2.normalize(hue, None, alpha=min_hue, beta=max_hue, norm_type=cv2.NORM_MINMAX)

    min_sat, max_sat = get_low_high(saturation)
    saturation = cv2.normalize(saturation, None, alpha=min_sat, beta=max_sat, norm_type=cv2.NORM_MINMAX)

    min_val, max_val = get_low_high(value)
    value = cv2.normalize(value, None, alpha=min_val, beta=max_val, norm_type=cv2.NORM_MINMAX)

    return hue, saturation, value

def get_low_high(image):
    mean = np.mean(image)
    std = np.std(image)
    low = max(mean - 2 * std, 0)
    high = min(mean + 2 * std, 255)
    return low, high

def resize_img(image, max_size):
    cols, rows = image.shape
    max_dim = max(rows, cols)

    if max_dim <= max_size:
        return (1, image)

    scale_factor = max_size / max_dim
    new_cols = int(cols * scale_factor)
    new_rows = int(rows * scale_factor)
    return (scale_factor, cv2.resize(image, (new_rows, new_cols)))

def compute_quad(polygon):
    min_x = np.min(polygon[:, 0])
    max_x = np.max(polygon[:, 0])
    min_y = np.min(polygon[:, 1])
    max_y = np.max(polygon[:, 1])

    step = 4

    top_left = min(polygon, key=lambda p: distance(p, (min_x, min_y)))
    top_right = min(polygon, key=lambda p: distance(p, (max_x, min_y)))
    bottom_right = min(polygon, key=lambda p: distance(p, (max_x, max_y)))
    bottom_left = min(polygon, key=lambda p: distance(p, (min_x, max_y)))

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


    orders = [
        [ "top_left", "top_right", "bottom_right", "bottom_left" ],
        [ "bottom_right", "bottom_left", "top_left", "top_right" ],
        [ "top_right", "top_left", "bottom_left", "bottom_right" ],
        [ "bottom_left", "bottom_right", "top_right", "top_left" ],
    ]

    for order in orders:
        top_left = (top_left[0], top_left[1] - step / 2)
        top_right = (top_right[0], top_right[1] - step / 2)
        bottom_right = (bottom_right[0], bottom_right[1] + step / 2)
        bottom_left = (bottom_left[0], bottom_left[1] + step / 2)

        for element in order:
            done = False
            if element == "top_left":
                while not done:
                    top_left = (top_left[0], top_left[1] + step)
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        top_left = (top_left[0], top_left[1] - step)
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True
            elif element == "top_right":
                while not done:
                    top_right = (top_right[0], top_right[1] + step)
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        top_right = (top_right[0], top_right[1] - step)
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True
            elif element == "bottom_right":
                while not done:
                    bottom_right = (bottom_right[0], bottom_right[1] - step)
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        bottom_right = (bottom_right[0], bottom_right[1] + step)
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True
            elif element == "bottom_left":
                while not done:
                    bottom_left = (bottom_left[0], bottom_left[1] - step)
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        bottom_left = (bottom_left[0], bottom_left[1] + step)
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True

    orders = [
        [ "bottom_left", "bottom_right", "top_right", "top_left" ],
        [ "top_right", "top_left", "bottom_left", "bottom_right" ],
        [ "bottom_right", "bottom_left", "top_left", "top_right" ],
        [ "top_left", "top_right", "bottom_right", "bottom_left" ],
    ]

    for order in orders:
        top_left = (top_left[0] - step / 2, top_left[1])
        top_right = (top_right[0] + step / 2, top_right[1])
        bottom_right = (bottom_right[0] + step / 2, bottom_right[1])
        bottom_left = (bottom_left[0] - step / 2, bottom_left[1])

        for element in order:
            done = False
            if element == "top_left":
                while not done:
                    top_left = (top_left[0] + step, top_left[1])
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        top_left = (top_left[0] - step, top_left[1])
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True
            elif element == "top_right":
                while not done:
                    top_right = (top_right[0] - step, top_right[1])
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        top_right = (top_right[0] + step, top_right[1])
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True
            elif element == "bottom_right":
                while not done:
                    bottom_right = (bottom_right[0] - step, bottom_right[1])
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        bottom_right = (bottom_right[0] + step, bottom_right[1])
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True
            elif element == "bottom_left":
                while not done:
                    bottom_left = (bottom_left[0] + step, bottom_left[1])
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        bottom_left = (bottom_left[0] - step, bottom_left[1])
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True

    orders = [
        [ "top_left", "top_right", "bottom_right", "bottom_left" ],
        [ "bottom_right", "bottom_left", "top_left", "top_right" ],
        [ "top_right", "top_left", "bottom_left", "bottom_right" ],
        [ "bottom_left", "bottom_right", "top_right", "top_left" ],
    ]

    for order in orders:
        top_left = (top_left[0], top_left[1] - step / 2)
        top_right = (top_right[0], top_right[1] - step / 2)
        bottom_right = (bottom_right[0], bottom_right[1] + step / 2)
        bottom_left = (bottom_left[0], bottom_left[1] + step / 2)

        for element in order:
            done = False
            if element == "top_left":
                while not done:
                    top_left = (top_left[0], top_left[1] + step)
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        top_left = (top_left[0], top_left[1] - step)
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True
            elif element == "top_right":
                while not done:
                    top_right = (top_right[0], top_right[1] + step)
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        top_right = (top_right[0], top_right[1] - step)
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True
            elif element == "bottom_right":
                while not done:
                    bottom_right = (bottom_right[0], bottom_right[1] - step)
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        bottom_right = (bottom_right[0], bottom_right[1] + step)
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True
            elif element == "bottom_left":
                while not done:
                    bottom_left = (bottom_left[0], bottom_left[1] - step)
                    quad = np.array([top_left, top_right, bottom_right, bottom_left])
                    if not Polygon(polygon).within(Polygon(quad)):
                        bottom_left = (bottom_left[0], bottom_left[1] + step)
                        quad = np.array([top_left, top_right, bottom_right, bottom_left])
                        done = True


    return np.array([top_left, top_right, bottom_right, bottom_left])

def default_perspective(image, quad):
    xmin, ymin = np.min(quad, axis=0)
    xmax, ymax = np.max(quad, axis=0)

    src_pts = np.array(
        [
            [xmin, ymin],
            [xmax - 1, ymin],
            [xmax - 1, ymax - 1],
            [xmin, ymax],
        ],
        dtype=np.float32,
    )


    output_height = int(ymax - ymin)
    output_width = int(xmax - xmin)

    dst_pts = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height],
        ],
        dtype=np.float32,
    )

    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return perspective_matrix, output_width, output_height

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

    return perspective_matrix, output_width, output_height

def blacken_mask(image, mask):
    mask_bool = mask.astype(bool)
    image = np.where(np.dstack([mask_bool] * 3), image, 0)
    return image


def distance(p1, p2):
    return ((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2) ** 0.5

def defill_holes(image, size=5):
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    closed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, morph_kernel)
    return closed_image
