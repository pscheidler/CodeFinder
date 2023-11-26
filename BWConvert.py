"""BWConvert Convert images to pure black and white

Interactively turn a color image into black and white so you can use that image for future symbol recognition
Initially copied from OpenCV help docs, heavily modified

Pressing 's' will save the image to saved_image.png, pressing 'q' will quit

"""

# Eyedropper says the HSV of the lines is at 20, 100, 18 to 22, 69, 20. HSV of the paper is 31, 32, 73 to 33, 50, 58
# This shows good detection with range of 11, 55, 149 to 43, 122, 243

import cv2

# Some static parameters
max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
save_file_name = 'saved_image.png'

# The functions below are linked to the trackbars and called when the values change
# They interpret the value and make sure we do not get max < min
def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv2.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv2.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv2.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv2.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv2.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv2.setTrackbarPos(high_V_name, window_detection_name, high_V)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
    parser.add_argument('-i', '--image', help='Specify an image, instead of using the camera.',
                        default=None, type=str)
    parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
    parser.add_argument('--grey', help='Use greyscale detection instead of HSB.', action="store_true")
    args = parser.parse_args()

    print("Pressing 's' will save the image to saved_image.png, pressing 'q' will quit")

    # Grab input from camera or a static image
    if args.image is None:
        cap = cv2.VideoCapture(args.camera)
    else:
        frame = cv2.imread(args.image)
        cap = None

    # Set up the 2 windows we will display
    cv2.namedWindow(window_capture_name)
    cv2.namedWindow(window_detection_name)

    # Set up track bars for detection parameters
    if args.grey:
        max_value_H = 255
    cv2.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
    if not args.grey:
        cv2.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
    cv2.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
    if not args.grey:
        cv2.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
    cv2.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
    if not args.grey:
        cv2.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)

    # If we're not getting the image from a camera, we can just read it in once here
    if cap is None:
        if args.grey:
            frame_input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_input = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    while True:
        # Read in camera image and convert it to desired format
        if cap is not None:
            ret, frame = cap.read()
            if frame is None:
                break
            if args.grey:
                frame_input = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_input = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Apply detection
        if args.grey:
            # frame_threshold = cv2.adaptiveThreshold(frame_HSV, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            #                            cv2.THRESH_BINARY_INV, 11, 2)
            _, frame_threshold = cv2.threshold(frame_input, low_H, low_S, cv2.THRESH_BINARY_INV)
        else:
            frame_threshold = cv2.inRange(frame_input, (low_H, low_S, low_V), (high_H, high_S, high_V))

        # display image. We could skip the frame input display for static images, but it is cheap and potentially
        # helpful to update it
        cv2.imshow(window_capture_name, frame_input)
        cv2.imshow(window_detection_name, frame_threshold)

        # Respond to key presses
        key = cv2.waitKey(30)
        if key == ord('s'):
            cv2.imwrite(save_file_name, frame_threshold)
        if key == ord('q') or key == 27:
            break
