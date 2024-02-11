import cv2
import numpy as np
import os

# print(os.getcwd)
img = cv2.imread(r"C:\Prasham\Productivity\RMI\Maze_Runner_Genesis\Assets\Flower.jpg")
resized = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
hsvImg = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
rgbImg = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
cv2.imshow("orignal", resized)
cv2.imshow("hsvImg", hsvImg)
cv2.imshow("grayscale", grayscale)
cv2.imshow("rgbImg", rgbImg)

cv2.waitKey(0)
cv2.destroyAllWindows