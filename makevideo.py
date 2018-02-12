"""
Kim Hua

Displays a video from the front camera (and records if desired). Recognizes
ArUco makers and displays the IDs and axes.

Revision History:
14 Nov 2017
    - First commit
1 Jan 2018
    - Refactored and cleaned up. Included user input for video title.
"""

import numpy as np
import cv2
from cv2 import aruco, imshow, waitKey, imwrite
import imutils
from imutils import resize
import sys
import threading as th
import speech_recognition as sr
from gtts import gTTS
import playsound
import pickle

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
fileName = 'values'
fileObject = open(fileName, 'r')
objectValues = pickle.load(fileObject)
cameraMatrix = objectValues[1]
rvecs = objectValues[3]
tvecs = objectValues[4]
retval = objectValues[0]
distCoeffs = objectValues[2]

book_count = 0
rx_count = 0
phone_count = 0


# obtain audio from the microphone
stop_event = th.Event()
ids = None



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

            if "see" in words:
                if ids is None:
                    print("No IDs are visible.")
                    say_shit("No IDs are found.")
                else:
                    
                    print("I see ___", ids)

            if "help" in words:
                found = False
                while found is False:
                if "keys" or "Medication" or "Book" in words:
                    print("It is in your frame")



def say_shit(shit):
    tts = gTTS(text=shit, lang='en')
    tts.save("shit.mp3")
    playsound.playsound("shit.mp3", True)

listener = th.Thread(target=listen_for_speech)
listener.start()

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

    if ids is not None:
        frame = aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 1.75, cameraMatrix, distCoeffs, rvecs, tvecs)
	    
        for i in range(ids.size):
            frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 1.75)

            """
            if 11 in ids:
                phone_count = phone_count + 1
            elif 11 not in ids: 
                phone_count = 0
            if phone_count == 15:
                print("Keys")

            if 6 in ids:
                rx_count = rx_count + 1
            elif 6 not in ids: 
                rx_count = 0
            if rx_count == 15:
                print("Medication")

            if 23 in ids:
                book_count = book_count + 1
            elif 23 not in ids: 
                book_count = 0
            if book_count == 15:
                print("Book")
            """
                
            

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
