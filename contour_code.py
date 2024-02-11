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
        
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        
        x = cv.getTrackbarPos('value','Trackbar') #getting trackbar value
        ret1, thresh1 = cv.threshold(gray, x, 255, cv.THRESH_BINARY)#binary thresholding
        
        contours, hierarchy = cv.findContours(thresh1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(frame, contours, -1, (0, 255, 0), 3)

        cv.imshow("Binary Thresholding",thresh1)
        cv.imshow("Contours",frame)
        
    if cv.waitKey(25) & 0xFF == ord('q'): #escape on entering q
        break
    
cap.release()
cv.destroyAllWindows()