import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def laplacian_filter(image_path):
    ddepth = cv.CV_16S
    kernel_size = 3

    # Load an image
    src = cv.imread(image_path, cv.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print('Error: Unable to load image:', image_path)
        return

    # Remove noise by blurring with a Gaussian filter
    src = cv.GaussianBlur(src, (3, 3), 0)

    # Convert the image to grayscale
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    # Apply Laplacian function
    dst = cv.Laplacian(src_gray, ddepth, kernel_size)

    # Converting back to uint8
    abs_dst = cv.convertScaleAbs(dst)

    # Display
    plt.subplot(121), plt.imshow(src, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(abs_dst, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

# Path to the folder containing images
folder_path = r'C:\Users\Silpaja\Videos\Kidney-Stone-Detection-IP-master\images'

# Process each image in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image (supports jpg, png, and jpeg)
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        # Construct the full path to the image
        image_path = os.path.join(folder_path, filename)

        # Process the image
        laplacian_filter(image_path)
