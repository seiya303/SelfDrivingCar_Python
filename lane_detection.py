from picamera.array import PiRGBArray
from picamera import PiCamera
import matplotlib.pyplot as plt  
import numpy as np
import time
import imutils
import cv2



camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 30, 110)
    return canny

def roi(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
    [(6, 327), (767, 327), (430, 4)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 =  line.reshape(4)
            cv2.line( line_image, (x1,y1), (x2,y2), (255, 0, 0), 5)
    return line_image

def avg_line(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

def detectImg(image)

    img = imutils.resize(image, width=1000)
    lane_image = np.copy(img)
    canny = canny(lane_image)
    roi = roi(canny)
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 10, np.array([]), minLineLength=30, maxLineGap=5)
    #print(lines)
    line_image = display_lines(lane_image, lines)
    mixed_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    averaged_lines = avg_line(image, lines)

    return averaged_lines

# plt.imshow(roi)
# plt.show()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    image = frame.array
    processed_image = detectImg(image)

    cv2.imshow("Frame", processed_image)
    key = cv2.waitKey(1) & 0xFF
    
    rawCapture.truncate(0)
    if key == ord("q"):
        break
