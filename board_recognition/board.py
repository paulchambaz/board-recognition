import board_recognition as br

import matplotlib.pyplot as mplt
import numpy as np

def get_board_polygon(image):
    
    mplt.imshow(image, cmap='gray')
    mplt.show()

    # print("Gaussian")
    gaussian = gaussian_smoothing(image, kernel_size=5, sigma=1.4)

    # mplt.imshow(gaussian, cmap='gray')
    # mplt.show()

    # print("Gradient")
    gradient = gradients(gaussian)

    # mplt.imshow(gradient, cmap='gray')
    # mplt.show()

    # print("Binary")
    binary_image = br.binarize(gradient, .2)

    # mplt.imshow(binary_image, cmap='gray')
    # mplt.show()

    # print("Invert")
    inverted_image = br.invert(binary_image)

    # mplt.imshow(inverted_image, cmap='gray')
    # mplt.show()

    # print("Connected components")
    connected_components = br.get_connected_components(inverted_image)
    largest_component = max(connected_components, key=len)
    size_threshold = 0.5 * len(largest_component)

    huge_components = [component for component in connected_components if len(component) >= size_threshold]

    if len(huge_components) == 1:
        connected_component = huge_components[0]
        print('selected because there is only one')
    else:
        bounding_boxes = [get_bounding_box(component) for component in huge_components]

        contained_components = []
        for i, box1 in enumerate(bounding_boxes):
            for j, box2 in enumerate(bounding_boxes):
                if i != j and is_inside(box1, box2, margin=32):
                    contained_components.append(huge_components[i])
                    break

        if len(contained_components) == 1:
            connected_component = contained_components[0]
            print('selected from bounding box')
        else:
            centroids = [get_centroid(component) for component in huge_components]

            target_x = image.shape[1] * .6
            target_y = image.shape[0] * .5
            connected_component = min(huge_components, key=lambda c: distance(get_centroid(c), (target_x, target_y)))
            # connected_component = min(huge_components, key=lambda c: abs(get_centroid(c)[1] - target_y))
            print('selected from proximity to target')

    component_image = br.create_component_image(image, connected_component)

    filled = fill_holes(component_image)
    inverted = 1.0 - filled
    inverted_filled = fill_holes(inverted)
    final = 1.0 - inverted_filled




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

def fill_holes(image, size=5):
    """Fill holes using a dilation + erosion method
    @param image The image to fill holes for
    @param size The size of the kernel used to dilate and erode (def 5)
    """
    kernel = np.ones((size, size), dtype=np.uint8)

    dilated_image = dilate(image, kernel)
    closed_image = erode(dilated_image, kernel)

    return closed_image

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

def erode(image, kernel):
    """Erodes the image
    @param image The image to erode for
    @parma kernel The kernel to use for erosion
    """
    padded_image = np.pad(image, kernel.shape[0] // 2, mode='constant', constant_values=1)
    eroded_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            eroded_image[i, j] = np.all(kernel == padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]])

    return eroded_image

def get_bounding_box(component):
    xs, ys = zip(*component)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    return x_min, x_max, y_min, y_max

def is_inside(inner_box, outer_box, margin=0):
    return (
        inner_box[0] + margin >= outer_box[0]
        and inner_box[1] - margin <= outer_box[1]
        and inner_box[2] + margin >= outer_box[2]
        and inner_box[3] - margin <= outer_box[3]
    )

def get_centroid(component):
    xs, ys = zip(*component)
    x_avg = sum(xs) / len(component)
    y_avg = sum(ys) / len(component)

    return x_avg, y_avg

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
