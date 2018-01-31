"""
Kim Hua

Calibrates the camera.

Revision History:
14 Nov 2017
	- First commit
"""

import numpy as np
from numpy.linalg import inv
import cv2
from cv2 import aruco, imshow, waitKey, imwrite
import imutils
from imutils import resize
import pickle
import sys
from variable_saver import save_function
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

num_markers_x = 5
num_markers_y = 7
board = aruco.GridBoard_create(num_markers_x, num_markers_y, 1.0, 0.1, aruco_dict)
board_img = board.draw((500, 700))
cv2.imwrite("board.png", board_img);

all_corners = []
all_ids = []
img_size = (0, 0)
num_frames = 0

cap = cv2.VideoCapture(0) # Default is 0. Change to different numbers
						  # if laptop has multiple cameras
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(True):
	ret, image = cap.read()
	if image is None:
		continue

	image = resize(image, width=800)

	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	corners, ids, rejected = aruco.detectMarkers(image, 
												 aruco_dict,
												 parameters=parameters)

	image_copy = image.copy()

	if ids is not None and len(ids) > 0:
		aruco.drawDetectedMarkers(image_copy, corners, ids)

	imshow("Frame", image_copy)

	key = waitKey(1)
	if key == ord('q'):
		break
	elif key == ord('c') and ids is not None and len(ids) > 0:
		print("Frame captured")
		all_corners.append(corners)
		all_ids.extend(ids)
		img_size = image.shape[:2]
		num_frames += 1
		# markerCounterPerFrame.append(len(ids))

	cv2.imshow('Frame', image)

retval, cameraMatrix, distCoeffs, rvecs, tvecs = aruco.calibrateCameraAruco(
	corners, 
	ids, 
	np.array([len(ids)], dtype=int), 
	# np.array([num_markers_x * num_markers_y]),
	board,
	img_size,
	None, 
	None)

save_function(retval, cameraMatrix, distCoeffs, rvecs, tvecs)
cap.release()
cv2.destroyAllWindows()