import cv2
import numpy as np

thresh_bar_name = "threshold"
main_window_name = "Code Image"
thresh_val = 0.7

def get_contours(gray, dilate: bool = False):
    if dilate:
        # each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 4))

        # Applying dilation on the threshold image
        # Warning! Not clear if this assignment will modify the original image!
        gray = cv2.dilate(gray, rect_kernel, iterations=1)

    # converting to Greyscale and thresholding should have already been done
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_matches_from_contour(contour, gray, dilate=True):
    if dilate:
        # each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Applying dilation on the threshold image
        # Warning! Not clear if this assignment will modify the original image!
        gray = cv2.dilate(gray, rect_kernel, iterations=1)

    x, y, w, h = cv2.boundingRect(contour)
    template = gray[y:y + h, x:x + w]
    # Compare Hu Moments with another contour's Hu Moments using a similarity metric

    locations = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(locations >= thresh_val)

    # Convert this to an array of x, y, w, h, so it is more compatible with other functions
    return_array = []
    for top_left in zip(*locations[::-1]):
        return_array.append([top_left[0], top_left[1], template.shape[1], template.shape[0]])
    return return_array


def point_in_rect(point_x, point_y, x, y, w, h) -> bool:
    return x <= point_x <= x+w and y <= point_y <= y+h


def get_contour_from_point(x_in, y_in, contours) -> int:
    """
    Find the matchin contour box that contains a point
    :param x, y: (x, y) point position
    :param contours: list of contours from findContours
    :return: Contour index or -1 for Not Found
    """
    for index,contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if point_in_rect(x_in, y_in, x, y, w,h):
            return index
    return -1


def box_contours(image, contours):
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue rectangle
    # Save the highlighted image
    # cv2.imwrite('highlighted_image.jpg', highlighted_image)


def mouse_callback(event, x, y, flags, param):
    global highlighted_image
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    print(f"Left mouse button clicked at ({x}, {y})")
    contour_index = get_contour_from_point(x, y, found_contours)
    if contour_index == -1:
        print(f"No contour found")
        return
    locations_xywh = get_matches_from_contour(found_contours[contour_index], gray, dilate=True)
    for rx, ry, rw, rh in locations_xywh:
        cv2.rectangle(highlighted_image, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 1)
    cv2.imshow(main_window_name, highlighted_image)


def thresh_trackbar(val):
    global thresh_val
    thresh_val = val/100
    # cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)


# Create an OpenCV window and set the mouse callback
cv2.namedWindow(main_window_name)
cv2.setMouseCallback(main_window_name, mouse_callback)

# Load the image
image = cv2.imread("Inputs/DetBwMod.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

found_contours = get_contours(gray)

# Create a copy of the image to draw the contours on
# highlighted_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
highlighted_image = image.copy()

box_contours(highlighted_image, found_contours)

cv2.createTrackbar(thresh_bar_name, main_window_name, 0, 100, thresh_trackbar)

cv2.imshow(main_window_name, highlighted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
