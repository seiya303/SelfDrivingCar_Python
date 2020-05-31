import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
# Same command function as streaming, its just now we pass in the file path, nice!
cap = cv2.VideoCapture('/Users/2018A00587/Desktop/embedded_system/sample3_rs2.mp4')

# FRAMES PER SECOND FOR VIDEO
fps = 15
WIDTH = 400
HEIGHT = 240

pts1 = np.array([[62,135],[340,135],[45,185],[362,185]],np.int32)
pts2 = np.array([[100,0],[280,0],[100,240],[280,240]],np.int32)

def Perspective(image, pts1): #원근법 변환
    """
    Capture a Region of Interest
    """
    # cv2.line(image,pt1=tuple(pts1[0]),pt2=tuple(pts1[1]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts1[1]),pt2=tuple(pts1[3]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts1[3]),pt2=tuple(pts1[2]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts1[2]),pt2=tuple(pts1[0]),color=(255,0,0),thickness=2)

    # cv2.line(image,pt1=tuple(pts2[0]),pt2=tuple(pts2[1]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[1]),pt2=tuple(pts2[3]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[3]),pt2=tuple(pts2[2]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[2]),pt2=tuple(pts2[0]),color=(255,0,0),thickness=2)

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

def Histogram(imgFinalDuplicate, imgFinalDuplicate1):
    histogramLane = np.uint8([])
    histogramLaneEnd = np.uint8([])
    
    for i in range(WIDTH):
        # ROILane = cv2.rectangle(imgFinalDuplicate,(i,140),(i+1,240),(0,0,255),3)
        # Matrix: rows 100, cols 1 
        ROILane= imgFinalDuplicate[HEIGHT-100:HEIGHT, i]
        # scaling (0 ~ 255) to (0 ~ 1)
        ROILane = cv2.divide(ROILane, 255)
        # summation of white pixels: append the number of white pixels in ROILane to hisogramLane  
        histogramLane = np.append(histogramLane, ROILane.sum(axis=0)[0:1].astype(np.uint8), axis=0)

    for i in range(WIDTH):
        # ROILane = cv2.rectangle(imgFinalDuplicate,(i,140),(i+1,240),(0,0,255),3)
        # Matrix: rows 100, cols 1 
        ROILaneEnd= imgFinalDuplicate1[:HEIGHT, i]
        # scaling (0 ~ 255) to (0 ~ 1)
        ROILaneEnd = cv2.divide(ROILaneEnd, 255)
        # summation of white pixels: append the number of white pixels in ROILane to hisogramLane  
        histogramLaneEnd = np.append(histogramLaneEnd, ROILaneEnd.sum(axis=0)[0:1].astype(np.uint8), axis=0)
    laneEnd = histogramLaneEnd.sum(axis=0)
    print(f"Lane END = {laneEnd}")
    return histogramLane, laneEnd


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
    cv2.imshow("position of white lanes", imgFinal)
    cv2.waitKey(1)

    Result = laneCenter - frameCenter
    return Result

# Always a good idea to check if the video was acutally there
# If you get an error at thsi step, triple check your file path!!
if cap.isOpened()== False: 
    print("Error opening the video file. Please double check your file path for typos. Or move the movie file to the same location as this script/notebook")
    
# While the video is opened
while cap.isOpened():
    
    # Read the video file.
    ret, frame = cap.read()
    
    # If we got frames, show them.
    if ret == True:
        imgPers = Perspective(frame, pts1)
        imgFinal, imgFinalDuplicate, imgFinalDuplicate1 = Threshold(imgPers)
        histogramLane, laneEnd = Histogram(imgFinalDuplicate, imgFinalDuplicate1)
        LeftLanePos, RightLanePos = LaneFinder(imgFinal, histogramLane)
        Result = LaneCenter(imgFinal, LeftLanePos, RightLanePos)

        if laneEnd > 15000:
            print("Lane End")

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

        if laneEnd > 15000:
            ss = "Lane End"
            cv2.putText(frame, ss, (1,50), 0, 1, color=(255,0,0), thickness=2)        

        elif Result == 0:
            ss = f"Result = {Result} Move Forward"
            cv2.putText(frame, ss, (1,50), 0, 1, color=(0,0,255), thickness=2)

        elif Result > 0:
            ss = f"Result = {Result} Move Right"
            cv2.putText(frame, ss, (1,50), 0, 1, color=(0,0,255), thickness=2)  
        
        elif Result < 0:
            ss = f"Result = {Result} Move Left"
            cv2.putText(frame, ss, (1,50), 0, 1, color=(0,0,255), thickness=2)  

         # Display the frame at same frame rate of recording
        # Watch lecture video for full explanation
        time.sleep(1/fps)
        cv2.imshow('frame',frame)
 
        # Press q to quit
        if cv2.waitKey(25) & 0xFF == ord('q'):           
            break
 
    # Or automatically break this whole loop if the video is over.
    else:
        break
        
cap.release()
# Closes all the frames
cv2.destroyAllWindows()