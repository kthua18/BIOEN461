"""
Kim Hua

Given a video and target time, this program will capture that frame with a
+/- 1 second buffer and save the frames in a pickle file.

Revision History:
3 Dec 2017
    - First commit
"""

import numpy as np
import cv2
from cv2 import aruco, imshow, waitKey, imwrite
import imutils
from imutils import resize
import pickle
from matplotlib import pyplot as plt
import sys

# This program takes in a video as input.
# Each frame is saved in a list 'frames'

# Calibration

# f = open('store.pckl', 'rb')
# frame_anal = pickle.load(f)
# # f.close()

video_title = 'output_rvec3.avi'
target_frame = 11 # Point in time you want to campture (seconds)
file_name = 'rvec5.pckl'

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

num_markers_x = 5
num_markers_y = 7
board = aruco.GridBoard_create(num_markers_x, num_markers_y, 1.0, 0.1, aruco_dict)
board_img = board.draw((500, 700))

# cv2.imwrite("board2.png", board_img);

cap = cv2.VideoCapture(video_title)
# out = cv2.VideoWriter('output_diagcube.avi',        # Uncomment all the "out"s to record
#     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G',),     # a new video.
#     15, 
#     (1280, 720))

all_corners = []
all_ids = []
img_size = (0, 0)
num_frames = 0
frames = []

target_start = target_frame * 15 - 15
target_end = target_frame * 15 + 15

############################ Calibration ############################
# while(True):
# 	ret, image = cap.read()
# 	if image is None:
# 		continue

# 	image = resize(image, width=800)

# 	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 	corners, ids, rejected = aruco.detectMarkers(image, 
# 												 aruco_dict,
# 												 parameters=parameters)

# 	image_copy = image.copy()

# 	if ids is not None and len(ids) > 0:
# 		aruco.drawDetectedMarkers(image_copy, corners, ids)

# 	imshow("Frame", image_copy)

# 	key = waitKey(1)
# 	if key == ord('q'):
# 		break
# 	elif key == ord('c') and ids is not None and len(ids) > 0:
# 		print("Frame captured")
# 		all_corners.append(corners)
# 		all_ids.extend(ids)
# 		img_size = image.shape[:2]
# 		num_frames += 1
# 		# markerCounterPerFrame.append(len(ids))

# retval, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraAruco(
# 	corners, 
# 	ids, 
# 	np.array([len(ids)], dtype=int), 
# 	#np.array([num_markers_x * num_markers_y]),
# 	board,
# 	img_size,
# 	None, 
# 	None)
#####################################################################
fileName = 'values.pckl'
fileObject = open(fileName, 'r')
objectValues = pickle.load(fileObject)
cameraMatrix = objectValues[1]
rvecs = objectValues[3]
tvecs = objectValues[4]
retval = objectValues[0]
distCoeffs = objectValues[2]
#retval = 1.094584583527# 2205

# cameraMatrix = np.array([[  1.17827738e+03,   0.00000000e+00,   1.86622709e+02],
#        [  0.00000000e+00,   1.13414958e+03,   3.47654849e+02],
#        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
#
# distCoeffs = np.array([[-0.01479509,  0.69136764,  0.00726789, -0.00518565, -1.881378  ]])
#
# rvecs = np.array([[ 2.2974803 ],
#        [ 2.07597976],
#        [ 0.25607078]])
#
# tvecs = np.array([[  0.68605559],
#        [ -4.74354269],
#        [ 17.78934125]])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # if frame is None:
    #     continue

    frame = resize(frame, width=1280) #, height=720)
    # out.write(frame)
    frames.append(frame)
 
	# lists of ids and the corners beloning to each id
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        # frame = aruco.drawDetectedMarkers(frame, corners, ids)
        
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 1.75, cameraMatrix, distCoeffs, rvecs, tvecs)
	    
        # for i in range(ids.size):
            # frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 1.75)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

f = open(file_name, 'wb')
pickle.dump(frames[target_start : target_end], f)
f.close()

