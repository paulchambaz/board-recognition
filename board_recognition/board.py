import board_recognition as br

import matplotlib.pyplot as mplt
import numpy as np

def get_board_polygon(image):

    # print("Gaussian")
    gaussian = gaussian_smoothing(image, kernel_size=5, sigma=1.4)

    # print("Gradient")
    gradient = gradients(gaussian)

    # print("Binary")
    binary_image = br.binarize(gradient, .2)

    # print("Invert")
    inverted_image = br.invert(binary_image)

    # print("Connected components")
    connected_components = br.get_connected_components(inverted_image)
    largest_component = max(connected_components, key=len)
    component_image = br.create_component_image(image, largest_component)
    
    # print("Fill holes")
    filled = fill_holes(component_image)

    mplt.imshow(filled, cmap='gray')
    mplt.show()

def gaussian_smoothing(image, kernel_size=3, sigma=1.0):
    """Applies a gaussian smoothing filter to the image
    @param image The image to smooth
    @param sigma The standard deviation of the gaussian
    @return The smoothed image
    """
    # get the kernel for the gaussian filter
    kernel = gaussian_kernel(kernel_size, sigma)
    # applying the convolution filter of the image and the gaussian kernel
    smoothed_image = br.convolution_filter(image, kernel)
    return smoothed_image

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
    # we define the horizontal and vertical Sobel kernels
    sobel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    sobel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
    # we compute the horizontal and vertical gradient images using convolution
    # filters
    gradient_x = br.convolution_filter(image, sobel_x)
    gradient_y = br.convolution_filter(image, sobel_y)

    # finally we compute the norm of the gradients
    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))

    return magnitude


def fill_holes(image):
    """Fills holes in a binary image
    @param image The binary image to process
    @return The image with filled holes
    """
    filled = np.zeros_like(image)
    for j in range(image.shape[0]):
        # find first pixel with a white value
        min = 0
        max = 0
        for i in range(0, image.shape[1]):
            if image[j, i] == 1.0:
                min = i
                break
        for i in range(image.shape[1] - 1, -1, -1):
            if image[j, i] == 1.0:
                max = i
                break
        for i in range(min, max):
            filled[j, i] = 1.0
    for i in range(image.shape[1]):
        # find first pixel with a white value
        min = 0
        max = 0
        for j in range(0, image.shape[0]):
            if image[j, i] == 1.0:
                min = j
                break
        for j in range(image.shape[0] - 1, -1, -1):
            if image[j, i] == 1.0:
                max = j
                break
        for j in range(min, max):
            filled[j, i] = 1.0

    return filled
