import cv2
import numpy as np

# Load the image
image = cv2.imread('plant_leaves.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Find contours in the binary image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour
for i, contour in enumerate(contours):
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crop the leaf from the original image
    leaf = image[y:y+h, x:x+w]
    
    # Save each leaf as a separate image
    cv2.imwrite(f'leaf_{i}.jpg', leaf)

# Display the original image with bounding boxes around the leaves
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Detected Leaves', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
