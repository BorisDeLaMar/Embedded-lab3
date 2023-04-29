import cv2
import numpy as np

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def process(frame):

    rect_size = 100

    width, height, channels = frame.shape
    start_point = (int(height/2 - rect_size/2), int(width/2 - rect_size/2))
    end_point = (int(height/2 + rect_size/2), int(width/2 + rect_size/2))
    color = (0, 0, 255)
    thickness = 2
    rect = cv2.rectangle(frame, start_point, end_point, color, thickness)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_sensivity = 15
    s_h = 255
    v_h = 255
    s_l = 50
    v_l = 50
    blue_upper = np.array([115 + h_sensivity, s_h, v_h])
    blue_lower = np.array([115 - h_sensivity, s_l, v_l])
    mask_frame = hsv_frame[start_point[1]:end_point[1] + 1, start_point[0]:end_point[0] + 1]
    mask_green = cv2.inRange(mask_frame, blue_lower, blue_upper)

    blue_rate = np.count_nonzero(mask_green)/(rect_size*rect_size)

    org = end_point
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
	
    if blue_rate > 0.9:
        text = cv2.putText(rect, ' blue ', org, font, fontScale, color, thickness, cv2.LINE_AA)
    else:
        text = cv2.putText(rect, ' not blue ', org, font, fontScale, color, thickness, cv2.LINE_AA)

    return rect

#Open Default Camera
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)

while(cap.isOpened()):
    #Take each Frame
    ret, frame = cap.read()

    processed = process(frame)
    # Show video
    cv2.imshow('Cam', processed)

    # Exit if "4" is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 52 : #ord 4
        #Quit
        print ('Good Bye!')
        break