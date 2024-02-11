import cv2 as cv
import numpy as np

#callback function for trackbar, plays no role
def nothing(x):
    pass

cap = cv.VideoCapture(0)

#window for trackbar
cv.namedWindow("Trackbar")
cv.createTrackbar('value','Trackbar',0,255,nothing)

while (cap.isOpened()):
    ret,frame=cap.read()
    if ret == True:
        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        cv.imshow("Grayscale Video",frame)
        x = cv.getTrackbarPos('value','Trackbar') #getting trackbar value
        ret1, thresh1 = cv.threshold(frame, x, 255, cv.THRESH_BINARY)#binary thresholding
        ret2, thresh2 = cv.threshold(frame, x, 255, cv.THRESH_BINARY_INV)#inverse binary thresholding
        cv.imshow("Binary Thresholding",thresh1)
        cv.imshow("Inverse Binary Thresholding",thresh2)
    if cv.waitKey(25) & 0xFF == ord('q'): #escape on entering q
        break
cap.release()
cv.destroyAllWindows()