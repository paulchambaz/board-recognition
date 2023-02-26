import board_recognition as br

import matplotlib.pyplot as mplt
import numpy as np

def get_board_polygon(image):
    print("Computing board polygon")
    canny_image = canny_algorithm(image)
    inverted_image = br.invert(canny_image)
    connected_components = br.get_connected_components(inverted_image)

    print(len(connected_components))
    for i in range(len(connected_components)):
        conponent_image = br.create_component_image(image, connected_components[i])
        mplt.imshow(conponent_image, cmap='gray')
        mplt.show()


def canny_algorithm(image):
    """Application of the Canny edge detection algoritm
    @param image The image to apply Canny to
    @return The image after Canny has been applied to it
    """
    # the canny detection is pretty complex, it takes five steps
    # 1. Apply Gaussian filter to smooth the image in order to remove the noise
    print("Computing gaussian")
    gaussian = gaussian_smoothing(image, kernel_size=5, sigma=1.4)
    # 2. Find the intensity gradients of the image
    print("Computing gradients")
    magnitude, direction = gradients(image)
    # 3. Apply gradient magnitude thresholding or lower bound cut-off
    # suppression to get rid of spurious response to edge detection
    print("Computing non magnitude thresholding")
    suppressed_image = non_maximum_suppression(magnitude, direction)
    # 4. Apply double threshold to determine potential edges
    # print("Computing double threshold")
    # threshold_image, strong_edges, weak_edges = double_threshold(suppressed_image, 0.05, 0.09)
    # 5. Track edge by hysteresis: Finalize the detection of edges by
    # suppressing all the other edges that are weak and not connected to strong
    # edges
    # final_image = edge_tracking(threshold_image, weak_edges, strong_edges)

    return magnitude

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
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # we compute the horizontal and vertical gradient images using convolution
    # filters
    gradient_x = br.convolution_filter(image, sobel_x)
    gradient_y = br.convolution_filter(image, sobel_y)

    # finally we compute the norm of the gradients
    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    # and the orientation
    orientation = np.arctan2(gradient_y, gradient_x)

    return magnitude, orientation

def non_maximum_suppression(magnitude, direction):
    # Quantize gradient direction to 0, 45, 90, or 135 degrees
    quantized_direction = np.round(direction / (np.pi / 4)) % 4

    # Pad the magnitude array with zeros
    padded_magnitude = np.pad(magnitude, 1, mode='constant')

    # Initialize output array
    suppressed = np.zeros_like(magnitude)

    # Loop over the image
    for j in range(1, magnitude.shape[0] + 1):
        for i in range(1, magnitude.shape[1] + 1):
            # Get the magnitude and direction of the current pixel
            mag = padded_magnitude[j, i]
            dir = quantized_direction[j-1, i-1]

            # Determine the neighbors along the gradient direction
            if dir == 0:
                neighbors = [padded_magnitude[j, i-1], padded_magnitude[j, i+1]]
            elif dir == 1:
                neighbors = [padded_magnitude[j-1, i+1], padded_magnitude[j+1, i-1]]
            elif dir == 2:
                neighbors = [padded_magnitude[j-1, i], padded_magnitude[j+1, i]]
            else:
                neighbors = [padded_magnitude[j-1, i-1], padded_magnitude[j+1, i+1]]

            # Suppress the current pixel if it is not the maximum among its neighbors
            if mag >= neighbors[0] and mag >= neighbors[1]:
                suppressed[j-1, i-1] = mag

    return suppressed

def double_threshold(image, low_threshold_ratio=0.05, high_threshold_ratio=0.15):
    # Calculate the high and low threshold values based on the ratios
    low_threshold = low_threshold_ratio * np.max(image)
    high_threshold = high_threshold_ratio * np.max(image)

    # Create a new image with the same shape as the input image
    output_image = np.zeros_like(image)

    # Set the pixel values of the output image based on the threshold values
    output_image[image >= high_threshold] = 255
    output_image[image < low_threshold] = 0

    # Find the weak edges and check if they are connected to strong edges
    weak_edges = (image >= low_threshold) & (image < high_threshold)
    strong_edges = (image >= high_threshold)
    indices = np.argwhere(weak_edges)
    for idx in indices:
        i, j = idx
        if (i > 0 and strong_edges[i - 1, j]) or \
           (i < image.shape[0] - 1 and strong_edges[i + 1, j]) or \
           (j > 0 and strong_edges[i, j - 1]) or \
           (j < image.shape[1] - 1 and strong_edges[i, j + 1]):
            output_image[i, j] = 255

    return output_image.astype(np.uint8), weak_edges, strong_edges


def edge_tracking(image, weak_edges, strong_edges):
    # Initialize an array to store the final edges
    edge_map = np.zeros_like(image)

    # Iterate through each weak edge pixel
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            if weak_edges[j, i] != 0:
                # Check if the weak edge pixel is connected to a strong edge pixel
                connected = False
                for dj in range(-1, 2):
                    for di in range(-1, 2):
                        # Check the neighboring pixels
                        if strong_edges[j+dj, i+di] != 0:
                            # Follow the edge in the direction of the gradient
                            # until it reaches another strong edge pixel
                            mag = image[j, i]
                            angle = np.arctan2(dj, di)
                            connected, mag = follow_edge(image, weak_edges,
                                                         strong_edges, j, i,
                                                         angle, mag)
                            if connected:
                                break
                    if connected:
                        break
                if connected:
                    edge_map[j, i] = mag

    return edge_map

def follow_edge(image, weak_edges, strong_edges, j, i, angle, mag):
    dj_new = int(np.round(np.sin(angle)))
    di_new = int(np.round(np.cos(angle)))
    while (j+dj_new >= 0 and j+dj_new < image.shape[0] and
       i+di_new >= 0 and i+di_new < image.shape[1]):
        if strong_edges[j+dj_new, i+di_new] != 0:
            return True, mag
        elif weak_edges[j+dj_new, i+di_new] != 0:
            mag_new = image[j+dj_new, i+di_new]
            if mag_new > mag:
                mag = mag_new
                dj_new = int(np.round(np.sin(angle)))
                di_new = int(np.round(np.cos(angle)))
            else:
                break
    return False, mag
