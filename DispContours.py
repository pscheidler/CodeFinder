import cv2
import numpy as np

# Load the image
image = cv2.imread("Inputs/DetBwMod.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply thresholding to separate symbols from the background
# ret, thresh = cv2.threshold(gray, 72, 238, cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 4))

# Applying dilation on the threshold image
dilation = cv2.dilate(gray, rect_kernel, iterations=1)

# Find contours in the thresholded image
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the image to draw the contours on
# highlighted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
highlighted_image = image.copy()

main_moment = cv2.HuMoments(cv2.moments(contours[1]))
# Compare Hu Moments with another contour's Hu Moments using a similarity metric


# Iterate through the contours
for index, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)

    this_moment = cv2.HuMoments(cv2.moments(contour))
    similarity = cv2.matchShapes(main_moment, this_moment, cv2.CONTOURS_MATCH_I1, 0)
    print(f"Similarity {index} = {similarity}")

    # Check if the symbol is large enough to be considered
    if w > 5 and h > 5:
        # Draw a rectangle around the contour
        sim_range = 0.7
        if index == 1:
            cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
        elif similarity >= sim_range:
            cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Green rectangle
        else:
            red_amt = int(255 * similarity / sim_range)
            cv2.rectangle(highlighted_image, (x, y), (x + w, y + h), (0, 255-red_amt, red_amt), 2)  # Red rectangle
    else:
        print(f"Ignoring contour {index}, wxh = {w}x{h}")

# Save the highlighted image
cv2.imwrite('highlighted_image.jpg', highlighted_image)

# You can also display the highlighted image using cv2.imshow() if needed
cv2.imshow('Highlighted Image', highlighted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
