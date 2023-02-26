import numpy as np

def grayscale(image):
    """Converts a numpy rgb image to grayscale
    @param image The image to convert
    @return The grayscale of the image
    """
    return np.mean(image, axis=2)

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
            window = padded_image[j : j + kernel_size, i : i + kernel_size]
            # then we do the sum of the two matrixes and and apply it to the
            # pixel at coordinate j, i
            output_image[j, i] = np.sum(window * kernel)
    return output_image

def invert(image):
    """Inverts an image
    @param image The image to invert
    @return The inverted image
    """
    inverted_image = 255 - image
    return inverted_image


def get_connected_components(img, connectivity=4):
    """
    Finds all connected components in a binary image
    @param img: A binary image as a NumPy array
    @param connectivity: The connectivity parameter (4 or 8)
    A list of all connected components in the format (x, y, value)
    """
    # Get the height and width of the image
    height, width = img.shape
    # Create a list to store the connected components
    components = []
    # Create a binary mask of the same size as the image
    mask = np.zeros((height, width), dtype=np.uint8)
    # Define the neighbors based on the connectivity parameter
    neighbors = get_neighbors(connectivity)
    components = []
    mask = np.zeros_like(img)
    lines, cols = img.shape
    for i in range(cols):
        for j in range(lines):
            if img[j, i] == 1 and mask[j, i] == 0:
                component = get_component(img, mask, neighbors, i, j)
                components.append(component)
    return components

def get_neighbors(connectivity):
    if connectivity == 4:
        return [(0, -1), (0, 1), (-1, 0), (1, 0)]
    elif connectivity == 8:
        return [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def get_component(img, mask, neighbors, i, j):
    lines, cols = img.shape
    component = []
    queue = [(i, j)]
    mask[j, i] = 1
    while queue:
        curr_j, curr_i = queue.pop(0)
        component.append((curr_i, curr_j))
        for dj, di in neighbors:
            neighbor_i, neighbor_j = curr_i + dj, curr_j + di
            if 0 <= neighbor_i < cols \
                    and 0 <= neighbor_j < lines \
                    and mask[neighbor_j, neighbor_i] == 0 \
                    and img[neighbor_j, neighbor_i] == 1:
                queue.append((neighbor_i, neighbor_j))
                mask[neighbor_j, neighbor_i] = 1
    return component

def create_component_image(image, component):
    new_image = np.zeros(image.shape, dtype=np.uint8)
    for pixel in component:
        new_image[pixel[1], pixel[0]] = 255
    return new_image
