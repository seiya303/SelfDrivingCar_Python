import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
# Same command function as streaming, its just now we pass in the file path, nice!
cap = cv2.VideoCapture('/Users/2018A00587/Desktop/embedded_system/sample3_rs.mp4')

# FRAMES PER SECOND FOR VIDEO
fps = 25

pts1 = np.array([[62,135],[340,135],[45,185],[362,185]],np.int32)
pts2 = np.array([[100,0],[280,0],[100,240],[280,240]],np.int32)

def Perspective(image, pts1): #원근법 변환
    """
    Capture a Region of Interest
    """
    # for i, pt in enumerate(pts1):
    #     cv2.line(image,pt1=tuple(pt), pt2=tuple(pts1[i-3]),color=(255,0,0),thickness=2)
    # cv2.polylines(image,[pts1],isClosed=True,color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts1[0]),pt2=tuple(pts1[1]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts1[1]),pt2=tuple(pts1[3]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts1[3]),pt2=tuple(pts1[2]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts1[2]),pt2=tuple(pts1[0]),color=(255,0,0),thickness=2)

    # cv2.line(image,pt1=tuple(pts2[0]),pt2=tuple(pts2[1]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[1]),pt2=tuple(pts2[3]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[3]),pt2=tuple(pts2[2]),color=(255,0,0),thickness=2)
    # cv2.line(image,pt1=tuple(pts2[2]),pt2=tuple(pts2[0]),color=(255,0,0),thickness=2)
    Matrix = cv2.getPerspectiveTransform(pts1.astype(np.float32), pts2.astype(np.float32))
    imgPers = cv2.warpPerspective(image, Matrix, (400, 240))
    return imgPers

def Threshold(imgPers):
    imgGray = cv2.cvtColor(imgPers, cv2.COLOR_RGB2GRAY)
    imgThresh = cv2.inRange(imgGray, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary threshold', imgThresh)
    cv2.waitKey(0)
    imgEdge = cv2.Canny(imgGray, 50, 200)
    imgFinal = cv2.add(imgThresh, imgEdge)
    imgFinal = cv2.cvtColor(imgFinal, cv2.COLOR_GRAY2RGB)
    imgFinalDuplicate = cv2.cvtColor(imgFinal, cv2.COLOR_RGB2BGR)
    imgFinalDuplicate1 = cv2.cvtColor(imgFinal, cv2.COLOR_RGB2BGR)
    return imgFinal, imgFinalDuplicate, imgFinalDuplicate1

def Histrogram(imgFinalDuplicate):
    histrogramLane = np.float32([])
    
    for i in range(400):
        # cv2.imshow('img',imgFinalDuplicate)
        ROILane = cv2.rectangle(imgFinalDuplicate,(i,140),(i+1,240),(0,0,255),3)
        ROILane = cv2.divide(ROILane, 255)
        print(ROILane.shape)
        print(ROILane.sum(axis=2))
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
        histrogramLane = Histrogram(imgFinalDuplicate)
        LeftLanePos, RightLanePos = LaneFinder(imgFinal, histrogramLane)
        Result = LaneCenter(imgFinal, LeftLanePos, RightLanePos)
    
        str(ss(f"Result = {Result}")) #StringStream
        cv2.putText(img, str(ss), [1,50], 0, 1, color=(0,0,255), thickness=2)        

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