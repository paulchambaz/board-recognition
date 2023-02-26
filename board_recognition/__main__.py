import board_recognition as br

import matplotlib.image as mimg
import numpy as np

def main():

    image = mimg.imread('data/0.jpg')

    gray = br.grayscale(image)

    br.get_board_polygon(gray)


