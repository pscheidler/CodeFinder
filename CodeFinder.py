import cv2
import numpy as np


def find_straight_lines(image_path, color_lower, color_upper, min_length):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresolding to separate symbols from the background
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize an empty list to store the detected lines
    detected_lines = []

    # Iterate through the contours
    for contour in contours:
        # Get the length of the contour
        length = cv2.arcLength(contour, True)

        # If the length meets the minimum length requirement
        if length >= min_length:
            # Fit a straight line to the contour
            vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)

            # Calculate the endpoints of the line
            left_x = int(x - vx * length)
            left_y = int(y - vy * length)
            right_x = int(x + vx * length)
            right_y = int(y + vy * length)

            detected_lines.append(((left_x, left_y), (right_x, right_y)))

    return detected_lines


# Specify the path to your image
image_path = "Inputs/DetReduced.png"

# Specify the color range (in HSV format) and minimum line length
color_lower = [0, 100, 100]  # Lower bound of the color range
color_upper = [10, 255, 255]  # Upper bound of the color range
min_length = 50  # Minimum line length

# Find and print the detected lines
lines = find_straight_lines(image_path, color_lower, color_upper, min_length)
for line in lines:
    print("Line: ", line)
