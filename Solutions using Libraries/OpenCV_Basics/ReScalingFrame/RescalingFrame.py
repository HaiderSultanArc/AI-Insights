import numpy as np
import cv2

cap = cv2.VideoCapture(0)  # VideoCapture captures the video, its arguments are Camera Index number (Simply the number of camera to be used, out of all connected ones) or name of Video file


# cap.set(self, propId, value), this method gets an ID (3 for Width and 4 for Height) and Value to set to that ID
def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)


def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)


def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)


def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)


def rescale_frame(frame, percent = 75):
    scale_percent = 75
    width = int(frame.shape[1] * scale_percent / 100) # frame.shape, returns a tuple of (row, columns and channels (if img is color), shape[1] here gives columns
    height = int(frame.shape[0] * scale_percent / 100) # frame.shape, returns a tuple of (row, columns and channels (if img is color), shape[0] here gives rows
    dim = (width, height) # dimensions
    return cv2.resize(frame, dim, interpolation= cv2.INTER_AREA) # cv2.resize() takes the image, desired dimension and a flag (interpolation), flag takes one of many inputs, in this case it is INTER_AREA


while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()  # read(self, image) , self = VideoCapture,  read() method reads the Video we are Capturing, it returns True in case of Correctly capturing of Video and False otherwise

    frame0 = rescale_frame(frame, percent=30)
    frame1 = rescale_frame(frame, percent=140)

    # Display the resulting frame
    cv2.imshow('frame0', frame0)
    cv2.imshow('frame1', frame1)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()