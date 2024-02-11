# -*- coding: utf-8 -*-

import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
filename = 'images2'

#Reading the maze file
img = cv2.imread(r"C:\Prasham\Productivity\RMI\Maze_Runner_Genesis\Assets\Maze.jpg") #Put the right path for the maze image
temp = np.copy(img)
cv2.imshow("Original" , img)

#Binary conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting maze image to grayscale image
cv2.imshow("gray", gray)

#Inverting thresholding will give us a binary image with a white wall and a black background.
#Theshold value for single channel, exactly 1/2 of the intensity
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("thresh", thresh)

# # # Contours - only binary images, to be detcted in white and background in balck
# Finding all contours in inverted thresholded image
# cv2.RETR_EXTERNAL: Retrieves only the external contours, or boundaries  of the objects in the image. It does not retrieve contours within contours.

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  

width = img.shape[1]
height = img.shape[0]

# draws the first contour in the thresholded image
contours1 = cv2.drawContours(thresh, contours, 0, (255, 255, 255), 5)  
cv2.imshow("contours1", contours1)

# hide the other larger contour
final_contour = cv2.drawContours(contours1, contours, 1, (0,0,0) , 5) 
cv2.imshow("final_contour", final_contour)

#Threshing it once more just to remove any white noise, we want lines whose white intesnty is really high
ret, thresh = cv2.threshold(final_contour, 240, 255, cv2.THRESH_BINARY) 
cv2.imshow("thresh" , thresh)

#Determining kernel size for erosion and dilation
#By trial and error (Making sure that the inner spaces are filled while dilating)
ke = 19
kernel = np.ones((ke, ke), np.uint8)

#Dilate
dilation = cv2.dilate(thresh, kernel, iterations=2)
cv2.imshow("dilation" , dilation)

#Erosion (PS: We are eroding the dilated image)
erosion = cv2.erode(dilation, kernel, iterations=1)
cv2.imshow("erosion" , erosion)

#Find differences between two images (So that we get the actual maze solving path)
diff = cv2.absdiff(dilation, erosion)
cv2.imshow("diff" , diff)

#To make the line thinner (We erode the difference again)
kernel1 = np.ones((5, 5), np.uint8)
img_erosion = cv2.erode(diff, kernel1, iterations=2)

cv2.imshow("img_erosion" , img_erosion)

#Skeletonize only works on binary values 0-1, so its necessary to do /255 before
ske = skeletonize(img_erosion/255)
plt.imshow(ske)
plt.show()

# Skeltonize outputs Boolean Values, True is 255 (white) and False is 0 (black)
# To get all True indexes from the ouutput of skeletonization
points = []
points_arg = []
for i in range(img.shape[0]): #rows = height (y axis)
    for j in range(img.shape[1]): #column = width (x axis)
        if ske[i][j] == True:
            #print(i,j)
            points.append(ske[j][i]) # simply returns True
            points_arg.append((j,i))  #PS : its (j,i)

#points_arg has (X,y) coordinates
X=[]
Y=[]

#To draw the solution and to seperate X and Y into deifferent lists
for i in range(len(points_arg)):
    img = cv2.circle(img ,(points_arg[i][0],points_arg[i][1]),1,(255,0,0),-1)
    cv2.imshow('undordered' , img) #Unordered nature of ouput of skeletonization
    # cv2.waitKey(5)
    X.append(points_arg[i][0])
    Y.append(points_arg[i][1])
    
cv2.imshow("semi_final" , img)

plt.scatter(X,Y)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows