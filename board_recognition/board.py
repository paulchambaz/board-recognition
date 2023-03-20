import board_recognition as br

import matplotlib.pyplot as mplt
import numpy as np
from tqdm import tqdm

def process_image(image):
    
    # print('original')
    # mplt.imshow(image, cmap='gray')
    # mplt.show()

    gaussian = gaussian_smoothing(image, kernel_size=5, sigma=1.4)

    # print("gaussian")
    # mplt.imshow(gaussian, cmap='gray')
    # mplt.show()

    gradient = gradients(gaussian)
    gradient = gaussian_smoothing(gradient, kernel_size=3, sigma=0.8)

    # print("gradient")
    # mplt.imshow(gradient, cmap='gray')
    # mplt.show()

    binary_image = br.binarize(gradient, .15)

    binary_image = fill_holes(binary_image, size=3)

    # print("binary")
    # mplt.imshow(binary_image, cmap='gray')
    # mplt.show()

    inverted_image = br.invert(binary_image)


#     mplt.imshow(inverted_image, cmap='gray')
#     mplt.show()

    # print("Connected components")
    connected_components = br.get_connected_components(inverted_image)
    largest_component = max(connected_components, key=len)
    size_threshold = 0.4 * len(largest_component)

    huge_components = [component for component in connected_components if len(component) >= size_threshold]

    # print('connected components')
    # for component in huge_components:
    #     component_image = br.create_component_image(image, component)
    #     mplt.imshow(component_image, cmap='gray')
    #     mplt.show()

    contained_components = []
    done = False
    if len(huge_components) == 1:
        connected_component = huge_components[0]
        # print('unique selection')
        done = True
    elif len(huge_components) == 2:
        bounding_boxes = [get_bounding_box(component) for component in huge_components]

        for i, box1 in enumerate(bounding_boxes):
            for j, box2 in enumerate(bounding_boxes):
                if i != j and is_inside(box1, box2, margin=32):
                    contained_components.append(huge_components[i])
                    break

        if len(contained_components) == 1:
            connected_component = contained_components[0]
            # print('bounding box selection')
            done = True

    if done == False:
            target_x = image.shape[1] * .5
            target_y = image.shape[0] * .6

            connected_component = min(huge_components, key=lambda c: distance(get_centroid(c), (target_x, target_y)))
            # print('target selection')

    component_image = br.create_component_image(image, connected_component)
    bbox_large = get_bounding_box(connected_component)
    bbox_large = (bbox_large[0], bbox_large[2], bbox_large[1], bbox_large[3])

    filled = fill_holes(component_image, size=5)
    inverted = 1.0 - filled
    inverted_filled = fill_holes(inverted, size=3)
    filled = 1.0 - inverted_filled

    flood_filled = mask_fill(filled)

    # print('filling')
    # mplt.imshow(filled, cmap='gray')
    # mplt.show()

    # mplt.imshow(flood_filled, cmap='gray')
    # mplt.show()

    return flood_filled

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

def mask_fill(mask):
    out = mask.copy()
    invert = 1.0 - out
    connected_components = br.get_connected_components(invert)

    for component in connected_components:
        component_image = br.create_component_image(mask, component)

        start_point = component[0]

        sleft = False
        for x in range(start_point[0], -1, -1):
            if out[start_point[1], x] == 1:
                sleft = True
                break

        sright = False
        for x in range(start_point[0], out.shape[1], 1):
            if out[start_point[1], x] == 1:
                sright = True
                break

        sup = False
        for y in range(start_point[1], -1, -1):
            if out[y, start_point[0]] == 1:
                sup = True
                break

        sdown = False
        for x in range(start_point[1], out.shape[0], 1):
            if out[y, start_point[0]] == 1:
                sdown = True
                break

        if sleft and sright and sup and sdown:
            for point in component:
                out[point[1], point[0]] = 1.0

    return out

# polygon algorithms

def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def circumcircle_radius(p1, p2, p3):
    a = euclidean_distance(p1, p2)
    b = euclidean_distance(p2, p3)
    c = euclidean_distance(p3, p1)
    
    # Check for collinear points
    if abs(a + b - c) < 1e-6 or abs(b + c - a) < 1e-6 or abs(c + a - b) < 1e-6:
        return np.inf
    
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
    if area == 0:
        return np.inf
    
    radius = (a * b * c) / (4 * area)
    return radius

def alpha_shape(points, alpha):
    num_points = len(points)
    triangles = []
    
    for i in range(num_points):
        for j in range(i + 1, num_points):
            for k in range(j + 1, num_points):
                p1, p2, p3 = points[i], points[j], points[k]
                radius = circumcircle_radius(p1, p2, p3)
                
                if radius <= alpha:
                    triangles.append([i, j, k])
    
    hull_indices = np.unique(np.array(triangles))
    return points[hull_indices]
