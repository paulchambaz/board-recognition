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
