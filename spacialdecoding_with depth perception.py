#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 00:52:59 2018

@author: tianqiu

Displays a video from the front camera (and records if desired). Recognizes
ArUco makers and displays the IDs and axes.

Revision History:
14 Nov 2017
    - First commit
1 Jan 2018
    - Refactored and cleaned up. INcluded user input for video title.
"""

import numpy as np
import cv2
from cv2 import aruco, imshow, waitKey, imwrite
import imutils
from imutils import resize
import sys

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()
width = 1280
height = 720
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
image = cv2.imread("image.png")
corners, ids, rejected = aruco.detectMarkers(image, aruco_dict, 
                                             parameters=parameters)
# Calibration for depth perception
w1 = corners[0][0][0][0]
w2 = corners[0][0][1][0]
l1 = corners[0][0][0][1]
l2 = corners[0][0][1][1]

PIXEL_WIDTH = (((w2 - w1)**2) + (l2 - l1)**2)**(1/2)

# initialize the known distance from the camera to the object
KNOWN_DISTANCE = 43.18

# initialize the known width of the marker
KNOWN_WIDTH = 4.445

FOCAL_LENGTH = (PIXEL_WIDTH * KNOWN_DISTANCE) / KNOWN_WIDTH

while True:
    record_video = input("Would you like to make a video? (Y/N) ")
    if record_video == 'Y' or record_video == 'y':
        video_title = input("What would you like to call your file? ")

        out = cv2.VideoWriter(video_title + '.avi', 
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G',), 
                              15, 
                              (width, height))

        break
    elif record_video == 'N' or record_video == 'n':
        break
    else:
        print("Sorry, I could not understand your answer.\n")
        continue
    object_interested = input("What would you like to find? (object's name)")

# Values for the labtop camera
retval = 1.1683438011882972

cameraMatrix = np.array([[  1.57388204e+03,   0.00000000e+00,   2.79199944e+02],
       [  0.00000000e+00,   1.56164329e+03,   4.48254997e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

distCoeffs = np.array([[ -9.72342061e-02,  -2.11149821e+01,  -1.86168913e-01,
          1.20994900e-01,   1.29739184e+02]])

rvecs = np.array([[-2.20829976],
       [ 2.18813097],
       [-0.46213975]])

tvecs = np.array([[ 10.20706288],
       [ -3.05142391],
       [ 42.93244523]])

# Operation    

while True:
    ret, frame = cap.read()
    if frame is None:
        continue

    frame_copy = frame.copy()
    frame = resize(frame, width=1280)
    if record_video == 'Y' or record_video == 'y':
            out.write(frame)
 
	# lists of ids and the corners beloning to each id
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    olddirection = 0

    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 1.75, cameraMatrix, distCoeffs, rvecs, tvecs)
        direction = 0
        for i in range(ids.size):
            frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 1.75)
            centre = corners[0][0][0][0]
            w1 = corners[0][0][0][0]
            w2 = corners[0][0][1][0]
            l1 = corners[0][0][0][1]
            l2 = corners[0][0][1][1]
            PIXEL_WIDTH = (((w2 - w1)**2) + (l2 - l1)**2)**(1/2)
            ACTUAL_DISTANCE = (KNOWN_WIDTH  *  FOCAL_LENGTH) / PIXEL_WIDTH
            print (ACTUAL_DISTANCE)
            x1 = 0.4218 * ACTUAL_DISTANCE + 500
            x2 = width - x1
            if centre < x1:
                direction = 3
                if olddirection != direction:
                    print ("Object", ids, "is located at your front left \n")
                    olddirection = direction
            elif centre < x2:
                direction = 2
                if olddirection != direction:
                    print ("Object", ids, "is located straight ahead \n")
                    olddirection = direction
            else:
                direction = 1
                if olddirection != direction:
                    print ("Object", ids, "is located at your front right \n")
                    olddirection = direction
                
    # Display the resulting frame
    cv2.imshow('Frame', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if record_video == 'Y' or record_video == 'y':
    out.release()
cv2.destroyAllWindows()