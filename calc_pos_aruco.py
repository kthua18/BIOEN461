"""
Kim Hua

Calculates the absolute position of axes and relative distance, using a 2x2 
inch cube with 1.75x1.75 inch ArUco markers.

Revision History:
2 Dec 2017
    - First commit
"""

import numpy as np
from numpy.linalg import inv
import cv2
from cv2 import aruco, imshow, waitKey, imwrite
import imutils
from imutils import resize
import sys

cameraMatrix = np.array([[  1.17827738e+03,   0.00000000e+00,   1.86622709e+02],
       [  0.00000000e+00,   1.13414958e+03,   3.47654849e+02],
       [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

rvecs = np.array([[[ 4.00352273,  1.32055063,  0.76692628]],

       [[-2.47778541, -0.04628945,  1.73957301]],

       [[-2.85156744,  0.17633705, -0.79549612]]])

tvecs = np.array([[[ 11.6511448 ,   3.45379932,  36.49432522]],

       [[  9.663749  ,   4.46517263,  32.95327419]],

       [[ 10.8556985 ,   4.31777307,  31.78238928]]])

corners = np.array([[[ 568.,  445.],
        [ 605.,  456.],
        [ 559.,  467.],
        [ 521.,  456.]]], dtype='f'), np.array([[[ 513.,  467.],
        [ 552.,  479.],
        [ 553.,  540.],
        [ 514.,  525.]]], dtype='f'), np.array([[[ 567.,  480.],
        [ 612.,  467.],
        [ 612.,  526.],
        [ 568.,  541.]]], dtype='f')

marker1_r = cv2.Rodrigues(rvecs[0])[0]
marker1_t = tvecs[0]
marker1_uv = corners[0][0][0]

uv = np.array([568.,  445., 1])
a = np.array([ 20538.95020463,
        16604.5541719,
           36.49432522])

b = uv - a

cameraInv = inv(cameraMatrix)
rInv = inv(marker1_r)

c = np.matmul(cameraInv, b)
d = np.matmul(rInv, c)
