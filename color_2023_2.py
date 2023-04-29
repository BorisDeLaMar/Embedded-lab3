import cv2
import numpy as np

class Color:
    def __init__(self, name, h_u, h_l):
        self.name = name
        self.h_u = h_u
        self.h_l = h_l
        self.s_u = 255
        self.v_u = 255
        self.s_l = 50
        self.v_l = 50

class Figure:
    def __init__(self, name, data):
        self.name = name
        self.data = data

class Rectang:
    def __init__(self, img_width, img_height, size):
        self.left_up = int((img_height - size)/2) #x1
        self.right_up = int((img_width - size)/2) #y1
        self.left_down = int((img_height + size)/2) #x2
        self.right_down = int((img_width + size)/2) #y2
        self.upscale_down = 0
        self.upscale_right = 0
        self.offset_down = 0
        self.offset_right = 0
        self.size = size
    def re_weight(self, img_width, img_height, upscale_down, upscale_right, offset_down, offset_right):
        self.upscale_down = upscale_down if (((img_height + self.size)/2 + upscale_down + self.offset_right < img_height) and ((self.size + upscale_down) > 0)) else self.upscale_down
        self.upscale_right = upscale_right if (((img_width + self.size)/2 + upscale_right + self.offset_down < img_width) and ((self.size + upscale_right) > 0)) else self.upscale_right
        self.offset_down = offset_down if (((img_width - self.size)/2 + offset_down > 0) and ((img_width + self.size)/2 + self.upscale_right + offset_down < img_width)) else self.offset_down
        self.offset_right = offset_right if (((img_height - self.size)/2 + offset_right > 0) and ((img_height + self.size)/2 + self.upscale_down + offset_right < img_height)) else self.offset_right
        self.left_up = int((img_height - self.size)/2 + self.offset_right)
        self.right_up = int((img_width - self.size)//2 + self.offset_down)
        self.right_down = int((img_width + self.size)/2 + self.upscale_right + self.offset_down)
        self.left_down = int((img_height + self.size)/2 + self.upscale_down + self.offset_right)
    def height(self):
        return int(abs(self.right_down - self.right_up)) if abs(self.right_down - self.right_up) > 0 else 1
    def width(self):
        return int(abs(self.left_down - self.left_up)) if abs(self.left_down - self.left_up) > 0 else 1

def hash(width, height, img):
    hash = ""
    for x in range(width):
        for y in range(height):
            val = img[y,x]
            if val==255:
                hash += "1"
            else:
                hash += "0"        
    return hash

def compare(hash1, hash2):
    count = 0
    l=len(hash1)
    for i in range(l):
        if hash1[i] == hash2[i]:
            count += 1
    return count

def optimal_borders_coeff(width, height):
    side = min(width, height)
    flag = True
    coeff = 1
    while(side/coeff > 60):
        side = side/coeff
        coeff += 1
    return coeff

def process(frame, rectang, colors, templates, i):
    
    colorit = (0, 0, 255)
    max_color_rate = max_count = 0
    colo, fig = 'none', ''
    
    thickness = 2
    rect = cv2.rectangle(frame, (rectang.left_up, rectang.right_up), (rectang.left_down, rectang.right_down), colorit, thickness)
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_frame = hsv_frame[rectang.right_up:rectang.right_down + 1, rectang.left_up:rectang.left_down + 1]

    coeff = optimal_borders_coeff(rectang.width(), rectang.height())
    r_width, r_height = int(rectang.width()/coeff), int(rectang.height()/coeff)

    resized_mask = cv2.resize(mask_frame, (r_width, r_height), interpolation = cv2.INTER_AREA)
    mask_gray = cv2.cvtColor(resized_mask, cv2.COLOR_BGR2GRAY)
    avg_mask = int(255/2) #mask_gray.mean()
    thres_mask, thresholded_mask = cv2.threshold(mask_gray, avg_mask, 255, 0)
    mask_hash = hash(r_width, r_height, thresholded_mask)
    
    for color in colors:
        color_upper = np.array([color.h_u, color.s_u, color.v_u])
        color_lower = np.array([color.h_l, color.s_l, color.v_l])
        mask_green = cv2.inRange(mask_frame, color_lower, color_upper)
        color_rate = np.count_nonzero(mask_green)/(rectang.width()*rectang.height())
        if max_color_rate < color_rate:
            max_color_rate = color_rate
            colo = color.name
            
    for template in templates:

        template.data = cv2.resize(template.data, (r_width, r_height), interpolation = cv2.INTER_AREA)
        templ_gray = cv2.cvtColor(template.data, cv2.COLOR_BGR2GRAY)
        avg_templ = templ_gray.mean()
        thres_tmpl, thresholded_templ = cv2.threshold(templ_gray, avg_templ, 255, 0)
        templ_hash = hash(r_width, r_height, thresholded_templ)

        cnt = compare(templ_hash, mask_hash)
        """if(i%100 == 0):
            print(template.name, cnt, sep = " ")"""
        if max_count < cnt:
            max_count = cnt
            fig = template.name
        """if(i%100 == 0):
            print(cnt, sep = ' ')"""
        cv2.imshow("Mask", thresholded_mask)
        cv2.imshow("Gray template", mask_gray)
        cv2.imshow("Template", thresholded_templ)
    """if(i%100 == 0):
        print("========")"""
    org = (rectang.left_up, rectang.right_down)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.7
	
    if max_color_rate > 0.55:
        #print(max_count, r_width*r_height*0.7, fig, sep = " ")
        if(max_count > r_width*r_height*0.75):
            colo += " " + fig
        text = cv2.putText(rect, colo, org, font, fontScale, colorit, thickness, cv2.LINE_AA)
    else:
        if(max_count > r_width*r_height*0.75):
            text = cv2.putText(rect, fig, org, font, fontScale, colorit, thickness, cv2.LINE_AA)
        else:
            text = cv2.putText(rect, 'none', org, font, fontScale, colorit, thickness, cv2.LINE_AA)

    return rect

#Open Default Camera
blue = Color('blue', 130, 100)
red = Color('red', 10, 0)
#red2 = Color('red', 355, 345)
green = Color('green', 75, 45)
colors = [blue, red, green]

triangle = Figure("triangle", cv2.imread("C:\\templates\\triangle.jpg"))
square = Figure("square", cv2.imread("C:\\templates\\square.jpg"))
recta = Figure("rectangle", cv2.imread("C:\\templates\\rectangle.jpg"))
templates = [triangle, square, recta]

"""color = np.uint8([[[0,255,0 ]]])
print(cv2.cvtColor(color, cv2.COLOR_BGR2HSV))"""

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
width, height, channels = frame.shape
rect_size = 100
rectang = Rectang(width, height, rect_size)

upscale_down = upscale_right = 0
offset_down = offset_right = 0
prev_width = prev_height = 0
i = 0
upscaling_down = upscaling_right = True
offsetting_down = offsetting_right = True

while(cap.isOpened()):
    #Take each Frame
    ret, frame = cap.read()
    i += 1

    width, height, channels = frame.shape
    
    if(prev_width != width or prev_height != height):
        rectang = Rectang(width, height, rect_size)
        prev_width, prev_height = width, height
    else:
        rectang.re_weight(width, height, upscale_down, upscale_right, offset_down, offset_right)

    processed = process(frame, rectang, colors, templates, i)
    # Show video
    cv2.imshow('Cam', processed)

    # Exit if "4" is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 52 : #ord 4
        #Quit
        print ('Good Bye!')
        break
    elif k == ord('k'):
        if(upscaling_down):
            if(rectang.upscale_down == upscale_down):
                upscale_down += 0.05*height
            else:
                upscale_down = rectang.upscale_down
        else:
            upscaling_down = True
            upscale_down += 0.05*height
    elif k == ord('j'):
        if(upscaling_right):
            if(rectang.upscale_right == upscale_right):
                upscale_right += 0.05*width
            else:
                upscale_right = rectang.upscale_right
        else:
            upscaling_right = True
            upscale_right += 0.05*width
    elif k == ord('i'):
        if(not upscaling_down):
            if(rectang.upscale_down == upscale_down):
                upscale_down -= 0.05*height
            else:
                upscale_down = rectang.upscale_down
        else:
            upscaling_down = False
            upscale_down -= 0.05*height
    elif k == ord('u'):
        if(not upscaling_right):
            if(rectang.upscale_right == upscale_right):
                upscale_right -= 0.05*width
            else:
                upscale_right = rectang.upscale_right
        else:
            upscaling_right = False
            upscale_right -= 0.05*width
    elif k == ord('w'):
        if(not offsetting_down):
            if(rectang.offset_down == offset_down):
                offset_down -= 0.02*height
            else:
                offset_down = rectang.offset_down
        else:
            offsetting_down = False
            offset_down -= 0.02*height
    elif k == ord('s'):
        if(offsetting_down):
            if(rectang.offset_down == offset_down):
                offset_down += 0.02*height
            else:
                offset_down = rectang.offset_down
        else:
            offsetting_down = True
            offset_down += 0.02*height
    elif k == ord('a'):
        if(not offsetting_right):
            if(rectang.offset_right == offset_right):
                offset_right -= 0.02*width
            else:
                offset_right = rectang.offset_right
        else:
            offsetting_right = False
            offset_right -= 0.02*width
    elif k == ord('d'):
        if(offsetting_right):
            if(rectang.offset_right == offset_right):
                offset_right += 0.02*width
            else:
                offset_right = rectang.offset_right
        else:
            offsetting_right = True
            offset_right += 0.02*width

cap.release()
cv2.destroyAllWindows()
