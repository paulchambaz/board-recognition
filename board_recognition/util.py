import matplotlib.pyplot as mplt
import math
import numpy as np

def preprocess(image):
    cols, rows, _ = image.shape
    max_dim = max(rows, cols)
    if max_dim > 512:
        scale_factor = 512 / max_dim
        new_cols = int(cols * scale_factor)
        new_rows = int(rows * scale_factor)
        resized = downsize_rgb(image, (new_cols, new_rows))
    else:
        resized = image

#     mplt.imshow(resized)
#     mplt.show()

    gray = grayscale(resized)

    # print('gray')

    # mplt.imshow(gray, cmap='gray')
    # mplt.show()

    contrasted = stretch_contrast(gray)

#     mplt.imshow(contrasted, cmap='gray')
#     mplt.show()

    sigmoid = 1 / (1 + np.exp(-4 * (contrasted - 0.5)))
    sigmoid = np.clip(sigmoid, 0, 1)

#     mplt.imshow(sigmoid, cmap='gray')
#     mplt.show()

    return sigmoid

def grayscale(image):
    """Converts a numpy rgb image to grayscale
    @param image The image to convert
    @return The grayscale of the image
    """
    return np.mean(image, axis=2).astype(np.uint8) / 255

def downsize_rgb(image, shape):
    """Resizes an image using bilinear interpolation
    @param image The image to resize
    @param shape The new shape of the image as a tuple (height, width)
    @return The resized image
    """
    # computes the ratio
    height, width, channels = image.shape
    new_height, new_width = shape

    x_ratio = width / new_width
    y_ratio = height / new_height

    new_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            x, y = int(j * x_ratio), int(i * y_ratio)
            x_weight, y_weight = (j * x_ratio) % 1, (i * y_ratio) % 1
            new_image[i, j] = (
                image[y, x] * (1 - x_weight) * (1 - y_weight)
                + image[y, min(x + 1, width - 1)] * x_weight * (1 - y_weight)
                + image[min(y + 1, height - 1), x] * (1 - x_weight) * y_weight
                + image[min(y + 1, height - 1), min(x + 1, width - 1)] * x_weight * y_weight
            ).astype(np.uint8)

    return new_image

def downsize_gray(image, shape):
    # computes the ratio
    height, width = image.shape
    new_height, new_width = shape

    x_ratio = width / new_width
    y_ratio = height / new_height

    new_image = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(new_height):
        for j in range(new_width):
            x, y = int(j * x_ratio), int(i * y_ratio)
            x_weight, y_weight = (j * x_ratio) % 1, (i * y_ratio) % 1
            new_image[i, j] = (
                image[y, x] * (1 - x_weight) * (1 - y_weight)
                + image[y, min(x + 1, width - 1)] * x_weight * (1 - y_weight)
                + image[min(y + 1, height - 1), x] * (1 - x_weight) * y_weight
                + image[min(y + 1, height - 1), min(x + 1, width - 1)] * x_weight * y_weight
            ).astype(np.uint8)

    return new_image

def stretch_contrast(image):
    """Stretch the dynamic range of a grayscale image to the full range [0,
    255]."""
    # Compute the minimum and maximum pixel values in the image
    min_val, max_val = np.min(image), np.max(image)
    # Scale the pixel values to the full [0, 255] range
    output_image = np.zeros_like(image)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            output_image[j, i] = (1.0 * (float(image[j, i]) - min_val) / (max_val - min_val))
    return output_image

def uint8_to_float(image):
    float_image = image.astype(np.float32) / 255.0
    return float_image

def binarize(image, threshold):
    """Binarize a grayscale image
    @param image The image to binarize
    @param threshold The threshold at which to binarize
    @return The binarized image
    """
    binary_image = image.copy()
    binary_image[binary_image < threshold] = 0.0
    binary_image[binary_image >= threshold] = 1.0
    return binary_image


def convolution_filter(image, kernel, mode='edge'):
    """Applies a convolution filter to a image
    @param image The input image
    @param kernel The convolution kernel to apply
    @param mode 'edge', 'constant', 'wrap', 'reflect'. Determines how the array
    borders are handled
    @return The output image of the convolution filter
    """
    kernel_size = kernel.shape[0]
    # we create a padding image, this is because a convolution filter applies
    # on outside the border of the initial image. Therefore, we fill it to make
    # sure the edges no not produce errors
    padding = kernel_size // 2
    padded_image = np.pad(image, padding, mode)
    # then we start applying the convolution filter
    output_image = np.zeros_like(image)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            # here we extract the window - a matrix of size [kernel_size x
            # kernel_size] from the padded image. since we get it from the
            # padded image we start at j, since 0 for image is j for the padded
            # image
            window = padded_image[j : j + kernel_size, i : i + kernel_size]\
                    .astype(np.float32)
            # then we do the sum of the two matrixes and and apply it to the
            # pixel at coordinate j, i
            output_image[j, i] = np.sum(window * kernel)
    return output_image

def invert(image):
    """Inverts an image
    @param image The image to invert
    @return The inverted image
    """
    inverted_image = 1.0 - image
    return inverted_image


def get_connected_components(image, connectivity=4):
    """
    Finds all connected components in a binary image
    @param image: A binary image as a NumPy array
    @param connectivity: The connectivity parameter (4 or 8)
    A list of all connected components in the format (x, y, value)
    """
    # we initialise the components list which we will return
    components = []
    # we create a mask - which will be used to keep track of pixels already
    # part of a comonent
    mask = np.zeros_like(image)
    # we compute the neighbors
    neighbors = get_neighbors(connectivity)
    # loops over the image row by row
    rows, cols = image.shape
    for j in range(rows):
        for i in range(cols):
            component = get_connected_component(i, j, image, mask, neighbors)
            if not len(component) == 0:
                # print("value:", image[j, i], "length of component:", len(component))
                components.append(component)
    return components

def get_neighbors(connectivity):
    if connectivity == 4:
        return [(0, -1), (0, 1), (-1, 0), (1, 0)]
    elif connectivity == 8:
        return [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]

def get_connected_component(i, j, image, mask, neighbors):
    """Finds all pixels member of a connected component
    @param image The image to search
    @param mask Keeps track of members that are already present in the
    component list
    @param neighbors The list of surrounding pixels to consider
    @param i The i coordinate of the pixel to search at
    @param j The j coordinate of the pixel to search at
    @param component The list which contains all components
    """
    component = []
    rows, cols = image.shape
    # we can't do this function in a recursive way because of python
    stack = [(i, j)]
    while stack:
        # pop the element
        i, j = stack.pop()
        # if we are outside the image or the current pixel is not part of the
        # list or if we have already visited that pixel
        if i < 0 or i >= cols or j < 0 or j >= rows \
                or image[j, i] == 0 or mask[j, i] == 1.0:
            # we stop no need to search for that pixel
            continue
        # if it is
        else:
            # then we add it to the list of members of the connected components
            component.append((i, j))
            mask[j, i] = 1.0
            # and we recursive search over all neighbors
            for di, dj in neighbors:
                stack.append((i + di, j + dj))
    return component

def create_component_image(image, component):
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for pixel in component:
        new_image[pixel[1], pixel[0]] = 1.0
    return new_image

def apply_hsv_modification(image):
    hsv_image = rgb_to_hsv(image)

    green_to_blue_mask = np.logical_and(hsv_image[..., 0] >= 60, hsv_image[..., 0] <= 260)
    medium_to_high_saturation_mask = hsv_image[..., 1] >= 0.1
    average_value_mask = np.logical_and(hsv_image[..., 2] >= 0.2, hsv_image[..., 2] <= 0.8)
    mask1 = np.logical_and(np.logical_and(green_to_blue_mask, medium_to_high_saturation_mask), average_value_mask)

    low_saturation_mask = hsv_image[..., 1] <= 0.6
    high_value_mask = hsv_image[..., 2] >= 0.3
    mask2 = np.logical_and(low_saturation_mask, high_value_mask)

    combined_mask = np.logical_or(mask1, mask2)
    mask = np.logical_not(combined_mask)

    hsv_image[mask, 2] = hsv_image[mask, 2] / 2

    return hsv_to_gray(hsv_image)

def rgb_to_hsv(rgb):
    # resize the array as 2d array of pixels so it's easier to manage
    input_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)

    r, g, b = rgb[:, 0] / 255., rgb[:, 1] / 255., rgb[:, 2] / 255.

    # get min and max for each r g b
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)

    # value is simply the max
    value = cmax

    # compute the delta betweeen pixels
    delta = cmax - cmin

    # saturation is delta / cmax when we can
    saturation = np.zeros_like(cmax)
    saturation[cmax == 0] = 0
    saturation[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]

    # hue has a pretty complex formula
    hue = np.zeros_like(value, dtype=np.float32)

    delta_zero_mask = delta == 0
    r_greater_mask = (r > g) & (r > b) & ~delta_zero_mask
    g_greater_mask = (g >= r) & (g > b) & ~delta_zero_mask
    b_greater_mask = (b >= r) & (b >= g) & ~delta_zero_mask

    hue[r_greater_mask] = 60.0 * ((g - b)[r_greater_mask] / delta[r_greater_mask] % 6.0)
    hue[g_greater_mask] = 60.0 * ((b - r)[g_greater_mask] / delta[g_greater_mask] + 2.0)
    hue[b_greater_mask] = 60.0 * ((r - g)[b_greater_mask] / delta[b_greater_mask] + 4.0)
    hue[delta_zero_mask] = 0.0

    # finally we create the hsv image and we return it at correct shape
    res = np.dstack([hue, saturation, value])
    return res.reshape(input_shape)

def hsv_to_gray(hsv):
    """Return a grayscale image from a hsv image
    @param hsv Numpy hsv image
    @return A grayscale image from the hsv image
    """
    return hsv[:, :, 2]

def create_point_image(image, component):
    image_rgb = np.stack((image,) * 3, axis=-1)
    for pixel in component:
        image_rgb[pixel[0], pixel[1]] = (1.0, 0.0, 0.0)
    return image_rgb
