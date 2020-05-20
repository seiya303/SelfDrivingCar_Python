

## PUT THIS ALL IN ONE CELL!

import cv2
import numpy as np

# Connects to your computer's default camera
cap = cv2.VideoCapture(0)


# Automatically grab width and height from video feed
# (returns float which we need to convert to integer for later on!)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(frame,25)

    gray = cv2.cvtColor(blur,cv2.COLOR_RGB2GRAY)

    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
    # Display the resulting frame
    print(ret)
    cv2.imshow('frame',thresh)
    
    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy the windows
cap.release()
cv2.destroyAllWindows()