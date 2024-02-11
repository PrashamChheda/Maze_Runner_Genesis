# Task objective: Following a specific colored object
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
pts = []
def nothing():
    pass
cv2.namedWindow("HSV", cv2.WINDOW_NORMAL)
cv2.createTrackbar("h", "HSV", 0, 180, nothing)
cv2.createTrackbar("s", "HSV", 0, 255, nothing)
cv2.createTrackbar("v", "HSV", 0, 255, nothing)
img = np.zeros((300, 300), np.uint8)

while True:

    ret, frame = cap.read()  
    print(frame.shape)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.getTrackbarPos("h", "HSV")
    s = cv2.getTrackbarPos("s", "HSV")
    v = cv2.getTrackbarPos("v", "HSV")
    lower = np.array([h, s, v]) # use the trackbars initially to choose the color you want
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernal = np.ones((15, 15), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255, 100, 0), 3)
    

    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)

    pts.append(list((cx, cy)))
    for i in pts:
        cv2.circle(frame, i, 7, (255, 0, 0), -1)

    cv2.imshow("", frame)
    # cv2.imshow("opening", opening)
    # cv2.imshow("img",img)
    # cv2.imshow("errosion", errosion)
    # cv2.imshow("mask", mask)
    # cv2.imshow("blur", blur)
    # cv2.imshow("hsv", hsv)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows
