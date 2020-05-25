import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np


def roi(image):
	# height = image.shape[0]
	# width = image.shape[1]
	polygons = np.array([
	[(6, 500), (767, 327), (430, 4)]
	])
	# pts1 = np.array([[203,350],[684,350],[0,500],[873,500]])
	# pts2 = np.array([[0,0],[1000,0],[0,750],[1000,750]])
	pts1 = np.float32([[40,135],[360,135],[5,185],[400,185]])
	pts2 = np.float32([[0,0],[400,0],[0,240],[400,240]])
	#mask = np.zeros_like(image)
	#cv2.fillPoly(mask, polygons, 255)
	#cv2.polylines(image,pts1,1,color=(255,0,0),thickness=2)
	#masked_image = cv2.bitwise_and(image, mask)
	Matrix = cv2.getPerspectiveTransform(pts1, pts2)
	imgPers = cv2.warpPerspective(image, Matrix, (400, 240))
	return imgPers

def Histrogram(imgclone):
    hist = np.float32([])
    hist = np.resize(hist, 400)
    
    for i in range(400):
        ROILane = imgclone(cv2.rectangle(i,140,1,100))
        ROILane = cv2.rectangle(imgclone, (i, 140), (1, 100), 255)
        ROILane = cv2.divide(255, ROILane)
        np.append(hist, int(sum(ROILane)[0]))
    return hist

def LaneFinder(imgclone, hist):
    LeftPtr = np.argmax(hist[:150])
    LeftLanePos = np.diff(hist[0], LeftPtr)

    RightPtr = np.argmax(hist[250:])
    RightLanePos = np.diff(hist[0], LeftPtr)

    cv2.line(imgFinal, [LeftLanePos, 0], [LeftLanePos, 240], color=(0,255,0), thickness=2)
    cv2.line(imgFinal, [RightLanePos, 0], [RightLanePos, 240], color=(0,255,0), thickness=2)
    return LeftLanePos, RightLanePos

def LaneCenter(imgclone, LeftLanePos, RightLanePos):
    laneCenter = (RightLanePos - LeftLanePos) / 2 + LeftLanePos
    frameCenter = 188

    cv2.line(imgFinal, [laneCenter, 0], [laneCenter, 240], color=(0,255,0), thickness=3)
    cv2.line(imgFinal, [frameCenter, 0], [frameCenter, 240], color=(255,0,0), thickness=3)

    Result = laneCenter - frameCenter
    return Result


image = cv2.imread('trialroad.jpg')
img = cv2.resize(image, (400, 240), interpolation = cv2.INTER_AREA)#imutils.resize(image, width=1000)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = cv2.inRange(gray, 188, 255)
blurred = cv2.GaussianBlur(gray, (5, 5), 50)
edge = cv2.Canny(blurred, 120, 150)
mixed = cv2.add(edge, mask)
mixed = imgclone = cv2.cvtColor(mixed, cv2.COLOR_RGB2BGR)
roi = roi(mixed)

#hist = Histrogram(roi)


# plt.imshow(roi)
# plt.show()
cv2.imshow("lanes", roi)
cv2.waitKey(0)