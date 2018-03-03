"""
Kim Hua

Displays a video from the front camera (and records if desired). Recognizes
ArUco makers and displays the IDs and axes.

Revision History:
14 Nov 2017
    - First commit
1 Jan 2018
    - Refactored and cleaned up. Included user input for video title.
3 Mar 2018
    - Implemented beeping for object finding
    - Added Google cloud services for TTS
    Known bugs:
    If lines 151-152 are uncommented, it will crash. Will revert from multithreading to
    multiprocessing. 
"""

import numpy as np
import math
import pyaudio
import cv2
from cv2 import aruco, imshow, waitKey, imwrite
import imutils
from imutils import resize
import sys
import threading as th
import speech_recognition as sr
from gtts import gTTS
import playsound
import time

# Set up Aruco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

# Set up USB camera
cap = cv2.VideoCapture(2) # To use laptop front cam, use 1. Back camera, use 2.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Values from pickle file
fileName = 'values'
fileObject = open(fileName, 'r')
objectValues = pickle.load(fileObject)
cameraMatrix = objectValues[1]
rvecs = objectValues[3]
tvecs = objectValues[4]
retval = objectValues[0]
distCoeffs = objectValues[2]

# Setting up tone generation
def sine(frequency, length, rate):
    length = int(length * rate)
    factor = float(frequency) * (math.pi * 2) / rate
    return np.sin(np.arange(length) * factor)


def play_tone(stream, pause, frequency=440, length=0.1, rate=44100):
    chunks = []
    chunks.append(sine(frequency, length, rate))

    chunk = np.concatenate(chunks) * 0.25

    stream.write(chunk.astype(np.float32).tostring())
    time.sleep(pause)

# Setting up voice recognition and TTS
def listen_for_speech():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        while not stop_event.is_set():
            print("Say something!")
            audio = r.listen(source) #, timeout=1)

            # recognize speech using Google Cloud
            try:
                print("Google thinks you said: " + r.recognize_google(audio))
                speech = r.recognize_google(audio)
            except sr.UnknownValueError:
                print("Google could not understand audio")
                continue
            except sr.RequestError as e:
                print("Google error; {0}".format(e))
                continue

            words = speech.split()

            # Season to fill this in with more sophisticated object tracking later
            if "see" in words:
                if ids is None:
                    print("No IDs are visible.")
                    say_shit("I do not see any IDs.")
                elif 6 in ids:
                    if corners[0][0][0][0] >= 500:
                        print("There are keys are to your right")
                        say_shit("There are keys to your right")
                    elif corners[0][0][0][0] < 780:
                        print("There are keys are to your left")
                        say_shit("There are keys to your left")
                    else:
                        print("There are keys are straight ahead")
                        say_shit("There are keys straight ahead")

def say_shit(shit):
    tts = gTTS(text=shit, lang='en')
    tts.save("shit.mp3")
    playsound.playsound("shit.mp3", True)

# Asking user if they'd like to record a video
while True:
    record_video = input("Would you like to make a video? (Y/N) ")
    if record_video == 'Y' or record_video == 'y':
        video_title = input("What would you like to call your file? ")

        out = cv2.VideoWriter(video_title + '.avi', 
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G',), 
                              15, 
                              (1280, 720))

        break
    elif record_video == 'N' or record_video == 'n':
        break
    else:
        print("Sorry, I could not understand your answer.\n")
        continue

# Values for Logitech HD Webcam C390
# retval = 1.0945845835272205

# cameraMatrix = np.array([[  1.17827738e+03,   0.00000000e+00,   1.86622709e+02],
#        [  0.00000000e+00,   1.13414958e+03,   3.47654849e+02],
#        [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])

# distCoeffs = np.array([[-0.01479509,  0.69136764,  0.00726789, -0.00518565, -1.881378  ]])

# rvecs = np.array([[ 2.2974803 ],
#        [ 2.07597976],
#        [ 0.25607078]])

# tvecs = np.array([[  0.68605559],
#        [ -4.74354269],
#        [ 17.78934125]])

# obtain audio from the microphone
stop_event = th.Event()
ids = None

# If you uncomment the below 2 lines, threading will fail you :(
# listener = th.Thread(target=listen_for_speech)
# listener.start()

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32,
                    channels=1, rate=44100, output=1)

# Begin video
while True:
    print("test4")
    ret, frame = cap.read()
    # if frame is None:
    #     continue

    frame_copy = frame.copy()
    frame = resize(frame, width=1280)

    # lists of ids and the corners beloning to each id
    corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 1.75, cameraMatrix, distCoeffs, rvecs, tvecs)
        
        for i in range(ids.size):
            frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 1.75)
            if record_video == 'Y' or record_video == 'y':
                 out.write(frame)

            # Beeps faster as the markerID=6 moves from left to right side of frame
            if 6 in ids:
                if corners[0][0][0][0] < 256:
                    play_tone(stream, 1)
                if corners[0][0][0][0] < 512 and corners[0][0][0][0] >= 256:
                    play_tone(stream, 0.8)
                if corners[0][0][0][0] < 768 and corners[0][0][0][0] >= 512:
                    play_tone(stream, 0.6)
                if corners[0][0][0][0] < 1024 and corners[0][0][0][0] >= 768:
                    play_tone(stream, 0.4)
                if corners[0][0][0][0] < 1280 and corners[0][0][0][0] >=1024:
                    play_tone(stream, 0.15)
            
    # Display the resulting frame
    cv2.imshow('Frame', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stop_event.set()
cap.release()
if record_video == 'Y' or record_video == 'y':
    out.release()
cv2.destroyAllWindows()
listener.join()

#mainloop(  )
