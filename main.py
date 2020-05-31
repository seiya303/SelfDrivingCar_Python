from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
# import motor
import numpy as np
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
# resolution 
camera.resolution = (400, 240)
# FR
camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(400, 240))
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera

WIDTH = 400
HEIGHT = 240

pts1 = np.array([[62,135],[340,135],[45,185],[362,185]],np.int32)
pts2 = np.array([[100,0],[280,0],[100,240],[280,240]],np.int32)

def Perspective(image, pts1): #원근법 변환
    """
    Capture a Region of Interest
    """
    cv2.line(image,pt1=tuple(pts1[0]),pt2=tuple(pts1[1]),color=(255,0,0),thickness=2)
    cv2.line(image,pt1=tuple(pts1[1]),pt2=tuple(pts1[3]),color=(255,0,0),thickness=2)
    cv2.line(image,pt1=tuple(pts1[3]),pt2=tuple(pts1[2]),color=(255,0,0),thickness=2)
    cv2.line(image,pt1=tuple(pts1[2]),pt2=tuple(pts1[0]),color=(255,0,0),thickness=2)

    # cv2.line(image,pt1=tuple(pts2[0]),pt2=tuple(pts2[1]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[1]),pt2=tuple(pts2[3]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[3]),pt2=tuple(pts2[2]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[2]),pt2=tuple(pts2[0]),color=(255,0,0),thickness=2)
    cv2.imshow("ROI", image)
    cv2.waitKey(1)
    
    Matrix = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
    imgPers = cv2.warpPerspective(image, Matrix, (WIDTH, HEIGHT))
    return imgPers

def Threshold(imgPers):
    imgGray = cv2.cvtColor(imgPers, cv2.COLOR_RGB2GRAY)

    # Blurring -----------------------------------------------
    # imgBlur = cv2.blur(imgPers, (5,5))
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 0)
    # imgBlur = cv2.medianBlur(imgPers, 5)
    # imgBlur = cv2.bilateralFilter(imgPers,9,75,75)
    # -----------------------------------------------

    imgThresh = cv2.inRange(imgBlur, 149, 255, cv2.THRESH_BINARY)

    # Erosion / Dilation-------------------------------------------------------
    kernel = np.ones((5,5),np.uint8)
    imgErode = cv2.erode(imgThresh,kernel,iterations=1) 
    # imgDilate = cv2.dilate(imgErode,kernel,iterations=1) 
    # -------------------------------------------------------

    sobel_horizontal = cv2.Sobel(imgErode, cv2.CV_8U, 1, 0, ksize=9)
    # kernel2 = np.ones((3,3),np.uint8)
    # imgMpl = cv2.morphologyEx(sobel_horizontal, cv2.MORPH_CLOSE, kernel2)
    
    cv2.imshow("threshold", sobel_horizontal)
    cv2.waitKey(1)
    
    # Canny edge detection : still need tuning
    imgEdge = cv2.Canny(imgGray, 270, 300)
    
    cv2.imshow("canny edge", imgEdge)
    cv2.waitKey(1)

    # combing the binary threshold and canny edge detector
    imgFinal = cv2.add(sobel_horizontal, imgEdge)
    imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_GRAY2RGB)
    imgFinalDuplicate = cv2.cvtColor(imgFinal, cv2.COLOR_RGB2BGR)
    imgFinalDuplicate1 = cv2.cvtColor(imgFinal, cv2.COLOR_RGB2BGR)
    return imgFinal, imgFinalDuplicate, imgFinalDuplicate1

def Histogram(imgFinalDuplicate):
    histogramLane = np.uint8([])
    
    for i in range(WIDTH):
        # ROILane = cv2.rectangle(imgFinalDuplicate,(i,140),(i+1,240),(0,0,255),3)
        # Matrix: rows 100, cols 1 
        ROILane= imgFinalDuplicate[HEIGHT-100:HEIGHT, i]
        # scaling (0 ~ 255) to (0 ~ 1)
        ROILane = cv2.divide(ROILane, 255)
        # summation of white pixels: append the number of white pixels in ROILane to hisogramLane  
        histogramLane = np.append(histogramLane, ROILane.sum(axis=0)[0:1].astype(np.uint8), axis=0)
    return histogramLane


def LaneFinder(imgFinal, histogramLane):
    # find the position of left edge of left lane
    LeftLanePos = np.argmax(histogramLane[:150])

    # find the position of left edge of right lane
    RightLanePos = 250 + np.argmax(histogramLane[250:])

    cv2.line(imgFinal, (LeftLanePos, 0), (LeftLanePos, HEIGHT), color=(0,255,0), thickness=2)
    cv2.line(imgFinal, (RightLanePos, 0), (RightLanePos, HEIGHT), color=(0,255,0), thickness=2)
    return LeftLanePos, RightLanePos

def LaneCenter(imgFinal, LeftLanePos, RightLanePos):
    laneCenter = ((RightLanePos - LeftLanePos) / 2 + LeftLanePos).astype(np.uint8)
    frameCenter = WIDTH // 2

    cv2.line(imgFinal, (laneCenter, 0), (laneCenter, HEIGHT), color=(0,255,0), thickness=3)
    cv2.line(imgFinal, (frameCenter, 0), (frameCenter, HEIGHT), color=(255,0,0), thickness=3)
    cv2.imshow("positions of white lanes", imgFinal)
    cv2.waitKey(1)
    
    Result = laneCenter - frameCenter
    return Result

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    imgPers = Perspective(image, pts1)
    imgFinal, imgFinalDuplicate, imgFinalDuplicate1 = Threshold(imgPers)
    histogramLane = Histogram(imgFinalDuplicate)
    LeftLanePos, RightLanePos = LaneFinder(imgFinal, histogramLane)
    Result = LaneCenter(imgFinal, LeftLanePos, RightLanePos)

    if Result == 0:
        print("Forward")

    elif 0 < Result < 10:
        print("Right1")

    elif 10 <= Result < 20:
        print("Right2")

    elif Result >= 20:
        print("Right3")

    elif -10 < Result < 0:
        print("Left1")

    elif -20 < Result <= -10:
        print("Left2")

    elif Result <= -20:
        print("Left3")

    ss = f"Result = {Result}" #StringStream
    cv2.putText(image, str(ss), (1,50), 0, 1, color=(0,0,255), thickness=2)

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    if key == ord("q"):
        break