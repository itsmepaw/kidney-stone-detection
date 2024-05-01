import cv2
import argparse
import imutils
import numpy as np
import matplotlib.pyplot as plt
import os

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path, 0)

    # Apply median blur
    dst = cv2.medianBlur(img, 5)

    # Calculate the Laplacian
    lap = cv2.Laplacian(dst, cv2.CV_64F)

    # Calculate the sharpened image
    sharp = dst - 0.3 * lap

    sharp = np.uint8(cv2.normalize(sharp, None, 0, 255, cv2.NORM_MINMAX))
    equ = cv2.equalizeHist(sharp)

    # Display
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(equ, cmap='gray')
    plt.title('Output Image'), plt.xticks([]), plt.yticks([])
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
        process_image(image_path)
