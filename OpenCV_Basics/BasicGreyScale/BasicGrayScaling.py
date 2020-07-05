import numpy as np
import cv2

cap = cv2.VideoCapture(0) # VideoCapture captures the video, its arguments are Camera Index number (Simply the number of camera to be used, out of all connected ones) or name of Video file

while True:
    ret, frame = cap.read() # read(self, image) , self = VideoCapture,  read() method reads the Video we are Capturing, it returns True in case of Correctly capturing of Video and False otherwise

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame0', frame) # this method is used to show an image in a window
    cv2.imshow('frame1', gray)  # this method is used to show an image in a window

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break