"""MatchSymbols

Run a combination of automated and manual pattern matching. This expects a black and white input image at
Inputs/DetBwMod.png. The symbols should be in white
"""
import cv2
import numpy as np
from enum import Enum
from ContourContainer import ContourContainer, ContourNotFound
from typing import Sequence


class MouseMode(Enum):
    MATCH = ord('m')
    DELETE = ord('d')
    ADD = ord('a')
    TRANSLATE = ord('t')
    SELECT = ord('s')
    GROUP = ord('g')
    UNGROUP = ord('u')
    NEW = ord('n')


thresh_bar_name = "threshold"
main_window_name = "Code Image"
thresh_val = 0.7
mouse_mode = MouseMode.MATCH
start_pos = [0, 0]
adding = False

def get_contours(gray, contours: ContourContainer, dilate: bool = False):
    if dilate:
        # each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 4))

        # Applying dilation on the threshold image
        # Warning! Not clear if this assignment will modify the original image!
        gray = cv2.dilate(gray, rect_kernel, iterations=1)

    # converting to Greyscale and thresholding should have already been done
    # Find contours in the thresholded image
    found_contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in found_contours:
        contours.add(contour=contour)



def get_matches_from_rect(rect: Sequence[int], gray, dilate=True):
    if dilate:
        # each word instead of a sentence.
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # Applying dilation on the threshold image
        # Warning! Not clear if this assignment will modify the original image!
        gray = cv2.dilate(gray, rect_kernel, iterations=1)

    x, y, w, h = rect
    template = gray[y:y + h, x:x + w]
    # Compare Hu Moments with another contour's Hu Moments using a similarity metric

    locations = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(locations >= thresh_val)

    # Convert this to an array of points at the center of each box
    match_array = []
    for top_left in zip(*locations[::-1]):
        match_array.append([top_left[0] + template.shape[1]//2, top_left[1] + template.shape[0]//2])
    return match_array


def box_contours(image, contours, color=(255, 0, 0)):
    for rect in contours:
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # Blue rectangle
    # Save the highlighted image
    # cv2.imwrite('highlighted_image.jpg', highlighted_image)


def match_from_point(x: int, y: int, contours: ContourContainer, det_image):
    print(f"Left mouse button clicked at ({x}, {y})")
    try:
        rect = contours.get_box(x_in=x, y_in=y)
    except ContourNotFound as e:
        print(f"No contour found\n{e.message}")
        return
    locations_xy = get_matches_from_rect(rect, det_image, dilate=True)
    # Clear all selections, then set the matching ones to Active
    contours.unselect_boxes(all=True)
    for x, y in locations_xy:
        contours.select_boxes(x_in=x, y_in=y)

def mouse_callback(event, x, y, flags, param):
    global contour_container
    global start_pos
    global adding
    if mouse_mode == MouseMode.MATCH:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        match_from_point(x, y, contour_container, gray)
    elif mouse_mode == MouseMode.DELETE:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        contour_index = contour_container.get_index_by_point(x, y)
        if contour_index == -1:
            print(f"No contour found")
            return
        contour_container.remove(contour_index)
    elif mouse_mode == MouseMode.ADD:
        if event == cv2.EVENT_LBUTTONDOWN:
            start_pos = [x, y]
            adding = True
            return
        if event != cv2.EVENT_LBUTTONUP:
            return
        adding = False
        rx = min(x, start_pos[0])
        ry = min(y, start_pos[1])
        w = abs(x - start_pos[0])
        h = abs(y - start_pos[1])
        print(f"Adding {[rx, ry, w, h]}")
        contour_container.add(rect=[rx, ry, w, h])
    elif mouse_mode == MouseMode.SELECT:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        try:
            index = contour_container.get_index_by_point(x, y)
        except ContourNotFound as e:
            print(f"No box at {[x,y]}")
            return
        contour_container.unselect_boxes(all=True)
        group = contour_container.get_group(index=index)
        if group == 0:
            return
        contour_container.select_boxes(group=group)
    elif mouse_mode == MouseMode.GROUP:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        active_group = contour_container.get_group(selected=True)
        this_index = contour_container.get_index_by_point(x, y)
        contour_container.set_group(active_group, index=this_index)
        contour_container.select_boxes(index=this_index)
    elif mouse_mode == MouseMode.UNGROUP:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        active_group = contour_container.get_group(selected=True)
        this_index = contour_container.get_index_by_point(x, y)
        if contour_container.get_group(index=this_index) != active_group:
            return
        contour_container.set_group(0, index=this_index)
        contour_container.unselect_boxes(index=this_index)
    elif mouse_mode == MouseMode.NEW:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        this_index = contour_container.get_index_by_point(x, y)
        if contour_container.get_group(index=this_index) != 0:
            return
        contour_container.set_group(contour_container.max_group+1, index=this_index)
        contour_container.unselect_boxes(all=True)
        contour_container.select_boxes(index=this_index)


def thresh_trackbar(val):
    global thresh_val
    thresh_val = val/100


def update_display(base_image, contours: ContourContainer):
    return_image = base_image.copy()
    boxes = contours.get_boxes(active=True)
    box_contours(return_image, boxes, color=(0, 255, 0))
    boxes = contours.get_boxes(active=False)
    box_contours(return_image, boxes, color=(255, 0, 0))
    return return_image

print(f"""
Welcome to the program that matches Symbols. This is intended to be used with a black and white image full of
mysterious symbols. When the image is opened, the software will automatically run contour matching and get an
initial list of symbols. Identified symbols are boxed in Blue, Active symbols are in Green
All commands are a single letter. Note that for many commands, you enter the letter then select the symbols to act on:
 z - Run automatic grouping
 l - Load saved settings
 x - eXport settings
 q - Quit
 a - Add box for symbol (note, this command requires you click and drag the shape of the box)
 d - Delete symbol (click in the symbol box)
 s - Select, click on the symbol and select all symbols it is grouped with
 g - Group, add selected symbols to the selected group
 u - Ungroup, remove selected symbols from the active group, to the unclassified group
 n - New group, select an ungrouped symbol and start a new group for it
 y - Show all ungrouped symbols
 m - Match, run the contour matching on the selected symbol. For debug only.
""")
# Create an OpenCV window and set the mouse callback
cv2.namedWindow(main_window_name)
cv2.setMouseCallback(main_window_name, mouse_callback)

# Load the image
base_image = cv2.imread("Inputs/DetBwMod.png")
# original_image = cv2.imread("Inputs/DetImage.jpg")

# Convert the image to grayscale
gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)

contour_container = ContourContainer(min_width=2, min_height=2)
get_contours(gray, contour_container)

# Create a copy of the image to draw the contours on

cv2.createTrackbar(thresh_bar_name, main_window_name, 0, 100, thresh_trackbar)

# cv2.imshow("Original", original_image)

while True:
    highlighted_image = update_display(base_image, contour_container)
    cv2.imshow(main_window_name, highlighted_image)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break
    elif key == ord('m'):
        mouse_mode = MouseMode.MATCH
    elif key == ord('d'):
        mouse_mode = MouseMode.DELETE
    elif key == ord('a'):
        mouse_mode = MouseMode.ADD
    elif key == ord('s'):
        mouse_mode = MouseMode.SELECT
    elif key == ord('g'):
        mouse_mode = MouseMode.GROUP
    elif key == ord('u'):
        mouse_mode = MouseMode.UNGROUP
    elif key == ord('n'):
        mouse_mode = MouseMode.NEW
    elif key == ord('x'):
        contour_container.save('settings.json')
        print(f"Saved!")
    elif key == ord('l'):
        contour_container.load('settings.json')
    elif key == ord('y'):
        contour_container.unselect_boxes(all=True)
        contour_container.select_boxes(group=0)
    elif key == ord('z'):
        print(f"Starting Match")
        for index in range(contour_container.length):
            print(f"{index=}")
            if contour_container.get_group(index=index) > 0:
                continue
            rect = contour_container.get_box(index=index)
            new_group = contour_container.max_group + 1
            matches = get_matches_from_rect(rect, gray)
            if len(matches) > contour_container.length:     # Probably garbage
                continue
            contour_container.set_group(new_group, index=index)
            for match in matches:
                match_index = contour_container.get_index_by_point(match[0], match[1])
                if match_index >= 0:
                    contour_container.set_group(new_group, index=match_index)
        print(f"Done Matching!")


cv2.destroyAllWindows()
