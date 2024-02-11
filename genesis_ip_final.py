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
    cv2.waitKey(5)
    X.append(points_arg[i][0])
    Y.append(points_arg[i][1])
    
cv2.imshow("semi_final" , img)

#plt.scatter(X,Y)
#plt.show()


#Logic to get the order right using distance 

min_arg = np.argmin(np.array(Y))  # returns index of the least term
ord_x = []
ord_y = []

x0 = X[min_arg]
y0 = Y[min_arg]

ord_x.append(x0)
ord_y.append(y0)

distance_list = []
X.pop(0) # Removes the first element 
Y.pop(0) # Removes the first element


for i in range(len(X)):
    min_dist = 10 #Setting an initial distance
    j_min = 0 #for getting the index
    for j in range(len(X)):
        x1 = X[j]  # first element after removing start element of array
        y1 = Y[j]  # first element after removing start element of array

        dist = (x1 - x0)**2 + (y1 - y0)**2
        if dist < min_dist:
            min_dist = dist
            x_next = x1
            y_next = y1
            j_min = j

    #Appending the new ordered next element to the ordered list
    ord_x.append(x_next)
    ord_y.append(y_next)
    
    #Removing the new ordered next element from prev array
    X.pop(j_min)
    Y.pop(j_min)
    
    #Setting the new ordered element as the new base value x0,y0
    x0 = x_next
    y0 = y_next

#To check if it is correct continous order
for i in range(len(ord_x)):
     b = cv2.circle(temp,(ord_x[i],ord_y[i]),1,(255,255,0),-1)
     cv2.imshow('bb', b)
     cv2.waitKey(5)


l_x = 14.7 #Actual width of maze (x-axis) in cms
ip_x= img.shape[1] #Columns = width (x-axis)
l_y = 13.7 # Actual height of maze  (y-axis) in cms
ip_y = img.shape[0] #Rows = height (y-axis)
l_5 = 7 #in cms
servo_to_centre = 13 #in cms

#Setting the scaling factor
c_x = l_x/ip_x
c_y = l_y/ip_y


#Setting the transforamtion matrix

t_r=[[0,c_y,0,(l_5/2 - (c_y*ip_y/2))],
     [c_x,0,0,(servo_to_centre - (c_x*ip_x/2))],
     [0,0,-1,0],
     [0,0,0,1]]



points_x=[]
points_y=[]

for i in range(len(ord_x)):
    points_x.append(ord_x[i])
    points_y.append(ord_y[i])

img_x=[]
img_y=[]

#Vertical stacking of x and y
p= np.vstack((points_x,points_y,np.zeros((1,len(points_x))),np.ones((1,len(points_x)))))
print(p)

#Dot product
mat = np.dot(t_r,p)

x = mat[0]
y = mat[1]

print(x)
print(y)
plt.scatter(x,y)
plt.show()

la = 14  # length of arm A in cm
lb = 20  # length of arm B in cm
lc = 6.7  # distance between
N_Points = len(x)

theta_1 = []
theta_4 = []
for i in range(0, N_Points):
    xp = x[i]
    yp = y[i]
 
    E1 = -2 * la * xp;
    E4 = 2 * la * (-xp + lc);

    F1 = -2 * la * yp;
    F4 = -2 * la * yp;

    G1 = la ** 2 - lb ** 2 + xp ** 2 + yp ** 2;
    G4 = lc ** 2 + la ** 2 - lb ** 2 + xp ** 2 + yp ** 2 - 2 * lc * xp;
    print(i, x[i] , y[i], N_Points)
    temp1 = math.sqrt(E1 ** 2 + F1 ** 2 - G1 ** 2);
    temp4 = math.sqrt(E4 ** 2 + F4 ** 2 - G4 ** 2);

    m1 = G1 - E1;
    m4 = G4 - E4;
#
    theta1_pos = 2 * math.atan((-F1 + temp1) / m1);
    theta1_neg = 2 * math.atan((-F1 - temp1) / m1);

    theta4_pos = 2 * math.atan((-F4 + temp4) / m4);
    theta4_neg = 2 * math.atan((-F4 - temp4) / m4);

    theta1_pos = math.degrees(theta1_pos);
    theta1_neg = math.degrees(theta1_neg);
    theta4_pos = math.degrees(theta4_pos);
    theta4_neg = math.degrees(theta4_neg);

    theta_1.append(theta1_pos)
    theta_4.append(theta4_neg)


theta_1_new = []
theta_4_new = []

for i in range(0, N_Points):
    if theta_1[i] < 0:
        temp = 1 * (theta_1[i] + 270);

    if theta_1[i] > 0:
        temp = theta_1[i] - 90;

    theta_1_new.append(temp)
    theta_4_new.append(theta_4[i] + 90)


####################################################################################
import serial
import time


N_Points = len(theta_4_new)
print(N_Points)
print("1")
# print(theta_4_new)
arm = serial.Serial('com7',9600)
# arm.write("163.13 145.19\n".encode('utf-8'))
# time.sleep(0.2)
# arm.write("67.67 156.49\n".encode('utf-8'))

print("Connected")
for i in range(0,N_Points):
  s = str(round(float(theta_1_new[i]),2))+" "+str(round(float(theta_4_new[i]),2))+"\n"
  print(s+" "+str(i))
  arm.write(s.encode('utf-8'))
  time.sleep(0.2)

cv2.waitKey(0)
cv2.destroyAllWindows()
