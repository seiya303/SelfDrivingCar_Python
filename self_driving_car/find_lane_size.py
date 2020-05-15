import matplotlib.pyplot as plt  
import numpy as np
import cv2

def canny(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	blur = cv2.GaussianBlur(gray, (5,5), 0)
	canny = cv2.Canny(blur, 30, 68)
	return canny

def roi(image):
	height = image.shape[0]
	width = image.shape[1]
	triangle = np.array([(0, 1055), (2609, 1055)], )

image = cv2.imread('road.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)

plt.imshow(canny)
plt.show()
# cv2.imshow("canny", canny)
# cv2.waitKey(0)
