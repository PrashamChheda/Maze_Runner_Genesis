import cv2
import numpy as np

# Read the image as a grayscale image
img = cv2.imread(r"C:\Prasham\Productivity\RMI\Maze_Runner_Genesis\Assets\S.jpg")
# cv2.imshow("image",img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image
ret,img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Step 1: Create an empty skeleton
size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

# Get a Cross Shaped Kernel
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

# Repeat steps 2-4
while True:
    #Step 2: Open the image
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    
    # cv2.imshow("Morphologised_image",open)
    #Step 3: Substract open from the original image
    temp = cv2.subtract(img, open)
    
    # cv2.imshow("tempImage",temp)
    
    #Step 4: Erode the original image and refine the skeleton
    eroded = cv2.erode(img, element)
    
    # cv2.imshow("erodedImage",eroded)
    
    skel = cv2.bitwise_or(skel,temp)
    
    cv2.imshow("Skeleton",skel)
    cv2.waitKey(100)
    
    img = eroded.copy()
    # cv2.imshow("image",img)
    # Step 5: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
    if cv2.countNonZero(img)==0:
        break

# Displaying the final skeleton
cv2.imshow("Skeleton",skel)
cv2.waitKey(0)
cv2.destroyAllWindows()