import numpy as np

def preprocess(image):
    gray = grayscale(image)
    cols, rows = gray.shape
    print("cols:", cols, "rows:", rows)
    max_dim = max(rows, cols)
    print("max_dim:", max_dim)
    if max_dim > 512:
        scale_factor = 512 / max_dim
        new_cols = int(cols * scale_factor)
        new_rows = int(rows * scale_factor)
        resized = downsize(gray, (new_cols, new_rows))
    else:
        resized = gray
    contrasted = stretch_contrast(resized)
    return uint8_to_float(contrasted)

def grayscale(image):
    """Converts a numpy rgb image to grayscale
    @param image The image to convert
    @return The grayscale of the image
    """
    return np.mean(image, axis=2)

def downsize(image, shape):
    """Resizes an image using bilinear interpolation
    @param image The image to resize
    @param shape The new shape of the image as a tuple (height, width)
    @return The resized image
    """
    # computes the ratio
    block_size_i = image.shape[0] // shape[0]
    block_size_j = image.shape[1] // shape[1]
    blocks = image[:shape[0] * block_size_i, :shape[1] * block_size_j]\
            .reshape(shape[0], block_size_i, shape[1], block_size_j)
    resized_image = np.mean(blocks, axis=(1, 3)).astype(np.uint8)
    return resized_image

def stretch_contrast(image):
    """Stretch the dynamic range of a grayscale image to the full range [0,
    255]."""
    # Compute the minimum and maximum pixel values in the image
    min_val, max_val = np.min(image), np.max(image)
    # Scale the pixel values to the full [0, 255] range
    output_image = np.zeros_like(image)
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            output_image[j, i] = (255 * (float(image[j, i]) - min_val) / (max_val - min_val)).astype(np.uint8)
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
