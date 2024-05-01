import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path, 0)

    # Apply Otsu's thresholding
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    # Apply watershed
    img_color = cv2.imread(image_path)
    img_color = cv2.medianBlur(img_color, 5)
    markers = cv2.watershed(img_color, markers)
    img_color[markers == -1] = [255, 0, 0]

    return img_color

# Path to the folder containing images
folder_path = r'C:\Users\Silpaja\Videos\Kidney-Stone-Detection-IP-master\images'

# Process each image in the folder
for filename in os.listdir(folder_path):
    # Check if the file is an image (supports jpg, png, and jpeg)
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        # Construct the full path to the image
        image_path = os.path.join(folder_path, filename)

        # Process the image
        processed_image = process_image(image_path)

        # Display the processed image
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.title('Processed Image')
        plt.xticks([]), plt.yticks([])
        plt.show()
