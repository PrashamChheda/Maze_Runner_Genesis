# playing with contours
import cv2
import numpy as np
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
# create a canvas
img = np.zeros((480,640,3), dtype=np.uint8)

while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    frame = cv2.flip(frame,1)
    #identifying the colour
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue_lower = np.array([150, 50, 50], np.uint8)
    blue_upper = np.array([180, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    kernel = np.ones((9, 9), "uint8")
    blue_mask = cv2.dilate(blue_mask, kernel)
    blue_mask = cv2.erode(blue_mask, kernel)

    #find the contour
    contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    global max_contour 
    if contours != ():
        #identifying the countour that is to be tracked
        max_contour = contours[0]
        for contour in contours:
            if cv2.contourArea(contour)>cv2.contourArea(max_contour):
                max_contour=contour
        #find the centre of target countour
        M = cv2.moments(max_contour)
        if M['m00'] != 0:

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.drawContours(frame, max_contour, -1, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
            cv2.putText(frame, "center", (cx - 20, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        
 
    cv2.imshow('frame', frame)
    cv2.imshow("canvas",img)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()