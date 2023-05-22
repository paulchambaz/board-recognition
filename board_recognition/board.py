import board_recognition as br

import matplotlib.pyplot as mplt
import numpy as np
from tqdm import tqdm
import cv2
import json

def get_board_polygon(image):
    pre_image = br.preprocess(image)
    polygon_image = br.process_image(pre_image) / 255
    point_indicies = np.argwhere(polygon_image == 1)
    point_indicies = point_indicies[::23]
    alpha_shape = br.alpha_shape(point_indicies, 0.0)
    alpha_shape_flipped = np.flip(alpha_shape, axis=1)
    return alpha_shape_flipped * image.shape[0] / pre_image.shape[0]

def process_image(image):
    # print('original')
    # mplt.imshow(image, cmap='gray')
    # mplt.show()

    gaussian = gaussian_smoothing(image, 5, 1.4)

    # print("gaussian")
    # mplt.imshow(gaussian, cmap='gray')
    # mplt.show()

    gradient = gradients(gaussian)
    gradient = gaussian_smoothing(gradient, 3, 0.8)

    # print("gradient")
    # mplt.imshow(gradient, cmap='gray')
    # mplt.show()

    # binary_image = br.binarize(gradient, .15)
    _, binary_image = cv2.threshold(gradient.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary_image = fill_holes(binary_image)
    # morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, morph_kernel)

    # print("binary")
    # mplt.imshow(binary_image, cmap='gray')
    # mplt.show()

    inverted_image = cv2.bitwise_not(binary_image)

    # mplt.imshow(inverted_image, cmap='gray')
    # mplt.show()

    _, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_image)

    stats = stats.tolist()
    stats = [(i,) + tuple(x) for i, x in enumerate(stats)]

    stats = sorted(stats[1:], key=lambda x: x[cv2.CC_STAT_AREA + 1], reverse=True)
    largest_area = stats[0][cv2.CC_STAT_AREA + 1]

    huge_components = []

    for stat in stats:
        label = stat[0]
        area = stat[cv2.CC_STAT_AREA + 1]
        if area >= largest_area * 0.4:
            component_image = np.zeros_like(inverted_image, dtype=np.uint8)
            component_image[labels == label] = 255
            huge_components.append(component_image)

    # print(len(huge_components))

    # for huge_component in huge_components:
    #     mplt.imshow(huge_component, cmap='gray')
    #     mplt.show()

    if len(huge_components) == 1:
        component_image = huge_components[0]
    elif len(huge_components) > 1:
        bounding_boxes = [get_bounding_box(component) for component in huge_components]

        contained_components = []
        for i in range(len(bounding_boxes)):
            for j in range(len(bounding_boxes)):
                if i != j and is_inside(bounding_boxes[i], bounding_boxes[j], margin=10):
                    contained_components.append(huge_components[i])
                    break

        if len(contained_components) == 1:
            component_image = contained_components[0]
        else: 
            target_x = image.shape[1] * .5
            target_y = image.shape[0] * .6

            component_image = min(huge_components, key=lambda c: distance(get_centroid(c), (target_x, target_y)))

    # mplt.imshow(component_image, cmap='gray')
    # mplt.show()

    filled = fill_holes(component_image, size=5)
    inverted = 1.0 - filled

    opened = fill_holes(inverted, size=3)
    filled = 1.0 - opened

    # mplt.imshow(filled, cmap='gray')
    # mplt.show()

    border_size = 1
    padded_image = np.pad(filled, pad_width=border_size, mode='constant', constant_values=0)

    edges = cv2.Canny(padded_image.astype(np.uint8), threshold1=30, threshold2=100)
    height, width = padded_image.shape
    frontier = edges[border_size:height+border_size, border_size:width+border_size]

    # mplt.imshow(frontier, cmap='gray')
    # mplt.show()

    return frontier

def get_frontier(image, size=3):
    kernel = np.ones((size, size), dtype=np.uint8)
    erode_image = erode(image, kernel, pad_value=0)
    return image - erode_image

def gaussian_smoothing(image, kernel_size=3, sigma=1.0):
    """Applies a gaussian smoothing filter to the image
    @param image The image to smooth
    @param sigma The standard deviation of the gaussian
    @return The smoothed image
    """
    smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return smoothed_image
    # # get the kernel for the gaussian filter
    # kernel = gaussian_kernel(kernel_size, sigma)
    # # applying the convolution filter of the image and the gaussian kernel
    # smoothed_image = br.convolution_filter(image, kernel)
    # return smoothed_image

def gaussian_kernel(size, sigma=1.0):
    """Creates the 2D Gaussian kernel which will be applied to the image for
    gaussian filtering
    @param size The size of the kernel
    @param sigma The standard deviation of the gaussian
    @return The 2D Gaussian kernel
    """
    # calculates the center pixel of the kernel
    center = size // 2
    # creates the initial kernel matrix
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # calculates the center
            x, y = i - center, j - center
            # the value of a pixel in the kernel is equals to value of the
            # gaussian at center (x, y) and standard deviation sigma
            kernel[i, j] = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # normalizes the kernel
    kernel = kernel / np.sum(kernel)
    return kernel

def gradients(image):
    """Computes the gradients of an image
    @param image The image to compute gradient for
    @return A tuple containing the magnitude and orientation image of the
    gradient
    """
    # we compute the horizontal and vertical gradient images using cv2.Sobel()
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # finally we compute the norm of the gradients
    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    return magnitude
    # # we define the horizontal and vertical Sobel kernels
    # sobel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    # sobel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    # # we compute the horizontal and vertical gradient images using convolution
    # # filters
    # gradient_x = br.convolution_filter(image, sobel_x)
    # gradient_y = br.convolution_filter(image, sobel_y)
    #
    # # finally we compute the norm of the gradients
    # magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    #
    # return magnitude

def fill_holes(image, size=5):
    """Fill holes using a dilation + erosion method
    @param image The image to fill holes for
    @param size The size of the kernel used to dilate and erode (def 5)
    """
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, morph_kernel)
    return closed_image
    # kernel = np.ones((size, size), dtype=np.uint8)
    #
    # dilated_image = dilate(image, kernel)
    # closed_image = erode(dilated_image, kernel)
    #
    # return closed_image

def dilate(image, kernel):
    """Dilates the image
    @param image The image to dilate for
    @parma kernel The kernel to use for dilation
    """
    padded_image = np.pad(image, kernel.shape[0] // 2, mode='constant', constant_values=0)
    dilated_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            dilated_image[i, j] = np.any(kernel == padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]])

    return dilated_image

def erode(image, kernel, pad_value=1):
    """Erodes the image
    @param image The image to erode for
    @parma kernel The kernel to use for erosion
    """
    padded_image = np.pad(image, kernel.shape[0] // 2, mode='constant', constant_values=pad_value)
    eroded_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            eroded_image[i, j] = np.all(kernel == padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]])

    return eroded_image

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
    # Find the centroid of the non-zero pixels
    moments = cv2.moments(image)
    centroid_x = moments['m10'] / moments['m00']
    centroid_y = moments['m01'] / moments['m00']

    return centroid_x, centroid_y

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def mask_fill(mask):
    out = mask.copy()

    # Convert the binary image to a format OpenCV can use
    out = (out * 255).astype(np.uint8)

    # Find contours in the binary image (Inverted)
    contours, _ = cv2.findContours(255 - out, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # We only consider contours with more than 1 points
        if len(contour) > 1:
            # cv2.isContourConvex checks if the contour is convex (convex contours are always simple closed shapes)
            if cv2.isContourConvex(contour):
                # cv2.contourArea calculates the area of the contour
                if cv2.contourArea(contour) > 0:
                    # cv2.fillPoly fills an area defined by several polygon
                    cv2.fillPoly(out, [contour], 255)

    # Normalize the output back to range [0, 1]
    out = out / 255.0
    return out
    # out = mask.copy()
    # invert = 1.0 - out
    # connected_components = br.get_connected_components(invert)
    #
    # for component in connected_components:
    #     component_image = br.create_component_image(mask, component)
    #
    #     start_point = component[0]
    #
    #     sleft = False
    #     for x in range(start_point[0], -1, -1):
    #         if out[start_point[1], x] == 1:
    #             sleft = True
    #             break
    #
    #     sright = False
    #     for x in range(start_point[0], out.shape[1], 1):
    #         if out[start_point[1], x] == 1:
    #             sright = True
    #             break
    #
    #     sup = False
    #     for y in range(start_point[1], -1, -1):
    #         if out[y, start_point[0]] == 1:
    #             sup = True
    #             break
    #
    #     sdown = False
    #     for x in range(start_point[1], out.shape[0], 1):
    #         if out[y, start_point[0]] == 1:
    #             sdown = True
    #             break
    #
    #     if sleft and sright and sup and sdown:
    #         for point in component:
    #             out[point[1], point[0]] = 1.0
    #
    # return out

import alphashape

def alpha_shape(points, alpha):
    alpha_shape = alphashape.alphashape(points, alpha)
    # alpha_shape = unary_union(alpha_shape)
    # alpha_shape = alphashape.alphashape(points, alpha)
    boundary_points = np.array(alpha_shape.exterior.coords).astype(np.uint32)
    return boundary_points

from PIL import Image, ImageDraw

def create_polygon_image(image, polygon):
    image_pil = Image.fromarray(image)
    size = image_pil.size

    image_polygon = Image.new("L", size, "black")
    draw = ImageDraw.Draw(image_polygon)
    board_polygon = list(map(tuple, polygon))
    draw.polygon(board_polygon, fill="white")
    return np.array(image_polygon)

def load_polygons_from_json(json_file_path):
    polygons = []
    try:
        f = open(json_file_path, 'r')
        json_data = json.load(f)
    except OSError:
        return polygons
    for shape in json_data['shapes']:
        polygons.append(shape['points'])
    return np.array(polygons[0]).astype(np.uint8)
