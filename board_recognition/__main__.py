import argparse
import matplotlib.image as mimg
import matplotlib.pyplot as mplt
import numpy as np

from .board import get_board
from .text import get_text

def main() -> None:
    parser = argparse.ArgumentParser(description="Get text from a board")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to the image")
    args = parser.parse_args()

    image = args.image
    
    image = mimg.imread(image)

    # poly_board, board = get_board(image.copy())

    text = get_text(image.copy())

    # fig, axs = mplt.subplots(3, 1, figsize=(10, 5)) 
    #
    # axs[0].imshow(image)
    # axs[0].set_title('Original Image')
    # axs[0].axis('off')
    #
    # axs[1].imshow(image)
    # overlay = np.zeros_like(image)
    # overlay[board == 255] = [0, 191, 255]
    # axs[1].imshow(overlay, alpha=0.5)
    # axs[1].set_title('Board')
    # axs[1].axis('off')
    #
    # axs[2].imshow(text, cmap='gray')
    # axs[2].set_title('Text')
    # axs[2].axis('off')
    #
    # mplt.tight_layout()
    # mplt.show()

if __name__ == "__main__":
    main()
