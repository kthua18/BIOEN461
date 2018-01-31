"""
Kim Hua

Calculates the absolute position of axes and relative distance, using a 2x2 
inch cube with 1.75x1.75 inch ArUco markers.

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
import math

def find_marker_pos(target_ids):
    target_pos = []
    for i in range(target_ids.size):
        target = np.where(marker_ids==ids[i])
        start = int((target[0]+1)*4 - 4)
        end = int((target[0]+1)*4)
        #print(pos_marker[start:end])
        target_pos = np.append(target_pos, pos_marker[start:end]) # pos_marker[start:end])

    #print(target_pos)
    target_pos = np.reshape(target_pos, (target_ids.size * 4, 3))

    return target_pos

def rotationMatrixToEulerAngles(r) :
 
    # assert(isRotationMatrix(R))
    R = cv2.Rodrigues(r)[0]

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z]) * 180 / math.pi

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

cap = cv2.VideoCapture('aortascan_clip.avi')    

all_corners = []
all_ids = []
img_size = (0, 0)
num_frames = 0
frames = []
frame_rate = 15

target_time1 = 2
target_frame1 = target_time1 * frame_rate

target_time2 = 31
target_frame2 = target_time2 * frame_rate

# Calibration values for LOGITECH C390 WEBCAM
fileName = 'values.pckl'
fileObject = open(fileName, 'r')
objectValues = pickle.load(fileObject)
cameraMatrix = objectValues[1]
rvecs = objectValues[3]
tvecs = objectValues[4]
retval = objectValues[0]
distCoeffs = objectValues[2]

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

m = 1.75
b = 0.25
z = 0

marker_ids = [1, 0]

frame_count = 0
prev_corners = []

pos_marker = np.float32([[z, z, z], # right markerID=1
    [m, z, z],
    [m, m, z],
    [z, m, z],
    [-b, z, -b-m],                  # left markerID=0
    [-b, z, -b],
    [-b, m, -b],
    [-b, m, -b-m]])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # if frame is None:
    #     continue

    frame = resize(frame, width=1280)
    frame_count = frame_count + 1
    frames.append(frame)
 
	# lists of ids and the corners beloning to each id
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    frame = aruco.drawDetectedMarkers(frame, corners, ids)
    prev_corners = corners

    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        target_pos = find_marker_pos(ids)

        _, rvecs, tvecs = cv2.solvePnP(target_pos, # pos_marker, 
            np.reshape(corners, (len(ids) * 4, 2)),
            cameraMatrix, 
            distCoeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE)

        if frame_count == target_frame1:
            print('First scan detected!')
            scan1_r = rvecs
            scan1_t = tvecs

        if frame_count == target_frame2:
            print('Second scan detected!')
            scan2_r = rvecs
            scan2_t = tvecs

        # rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 1.75, cameraMatrix, distCoeffs, rvecs, tvecs)
	    
        frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs, tvecs, 1.75)

    # Display the resulting frame
    cv2.imshow('Frame', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

dist = np.linalg.norm(scan1_t - scan2_t)
angle1 = rotationMatrixToEulerAngles(scan1_r)
angle2 = rotationMatrixToEulerAngles(scan2_r)
angle_diff = angle1 - angle2
print('Probe translation: ', dist, " inches")
print('Probe rotation:  ', angle_diff, " degrees")



