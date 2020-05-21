<<<<<<< HEAD
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import motor
import numpy as np
# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (400, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(400, 240))
# allow the camera to warmup
time.sleep(0.1)
# capture frames from the camera

pts1 = np.int32([[40,135]],[360,135],[0,185],[400,185]])
pts2 = np.int32([[100,0],[280,0],[100,240],[280,240]])

def Perspective(image): #원근법 변환
    """
    Capture a Region of Interest
    """
    cv2.polylines(image,[src],isClosed=True,color=(255,0,0),thickness=2)

    Matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgPers = cv2.warpPerspective(image, Matrix, Size(400, 240))
    return imgPers

def Threshold(imgPers):
    imgGray = cv2.cvtColor(imgPers, cv2.COLOR_RGB2GRAY)
    imgThresh = cv2.inRange(imgGray, 230, 255)
    imgEdge = cv2.Canny(imgGray, 900, 900, 3, False)
    imgFinal = cv2.add(imgThresh, imgEdge)
    imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_GRAY2RGB)
    imgFinalDuplicate = cv2.cvtColor(imgFinal, cv2.COLOR_RGB2BGR)
    imgFinalDuplicate1 = cv2.cvtColor(imgFinal, cv2.COLOR_RGB2BGR)
    return imgFinal, imgFinalDuplicate, imgFinalDuplicate1

def Histrogram(imgFinalDuplicate):
    histrogramLane = np.float32([])
    
    for i in range(400):
        ROILane = imgFinalDuplicate(cv2.rectangle(i,140,1,100))
        ROILane = cv2.divide(255, ROILane)
        np.append(histrogramLane, int(sum(ROILane)[0]))
    return histrogramLane


def LaneFinder(imgFinal, histrogramLane):
    LeftPtr = np.argmax(histrogramLane[:150])
    LeftLanePos = np.diff(histrogramLane[0], LeftPtr)

    RightPtr = np.argmax(histrogramLane[250:])
    RightLanePos = np.diff(histrogramLane[0], LeftPtr)

    cv2.line(imgFinal, [LeftLanePos, 0], [LeftLanePos, 240], color=(0,255,0), thickness=2)
    cv2.line(imgFinal, [RightLanePos, 0], [RightLanePos, 240], color=(0,255,0), thickness=2)
    return LeftLanePos, RightLanePos

def LaneCenter(imgFinal, LeftLanePos, RightLanePos):
    laneCenter = (RightLanePos - LeftLanePos) / 2 + LeftLanePos
    frameCenter = 188

    cv2.line(imgFinal, [laneCenter, 0], [laneCenter, 240], color=(0,255,0), thickness=3)
    cv2.line(imgFinal, [frameCenter, 0], [frameCenter, 240], color=(255,0,0), thickness=3)

    Result = laneCenter - frameCenter
    return Result

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    image = frame.array

    imgPers = Perspective(image)
    imgFinal, imgFinalDuplicate, imgFinalDuplicate1 = Threshold(imgPers)
    histrogramLane = Histrogram(imgFinalDuplicate)
    LeftLanePos, RightLanePos = LaneFinder(imgFinal, histrogramLane)
    Result = LaneCenter(imgFinal, LeftLanePos, RightLanePos)

    ss.str(f"Result = {Result}") #StringStream
    cv2.putText(img, str(ss), [1,50], 0, 1, color=(0,0,255), thickness=2)

    # show the frame
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    if key == ord("q"):
        break
=======
>>>>>>> cfddcfbde567f7c29c2428996d8023e6936d9d9d
