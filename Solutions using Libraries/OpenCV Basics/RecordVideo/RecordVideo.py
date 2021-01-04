import numpy as np
import cv2, os


filename = 'VideoCapture.avi' # .avi and .mp4 formats are better at working over different Systems
frames_per_seconds = 24.0 # how many fps we want our video to get recorded in
my_res = '720p' # resolution


def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)


# Standard Video Dimensions
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1280),
    "4k": (3840, 2160)
}

def get_dims(cap, res='1080p'):
    if (res in STD_DIMENSIONS):
        width, height = STD_DIMENSIONS[res]
    else:
        width, height = STD_DIMENSIONS['480p']

    change_res(cap, width, height)
    return width, height


VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'), # XVID is the Video Codec that we are using, it is in fourcc
    'mp4': cv2.VideoWriter_fourcc(*'XVID') # XVID is the Video Codec that we are using, it is in fourcc
}


def get_video_type(filename):
    filename, ext = os.path.splitext(filename)

    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    else:
        return VIDEO_TYPE['avi']


cap = cv2.VideoCapture(0)  # VideoCapture captures the video, its arguments are Camera Index number (Simply the number of camera to be used, out of all connected ones) or name of Video file
dims = get_dims(cap, res=my_res)
video_type_cv2 = get_video_type(filename)

out = cv2.VideoWriter(filename, video_type_cv2, frames_per_seconds, dims) # VideoWriter(), takes filename, Video Type, FPS and Dimensions as Parameter

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()  # read(self, image) , self = VideoCapture,  read() method reads the Video we are Capturing, it returns True in case of Correctly capturing of Video and False otherwise

    out.write(frame) # cv2.VideoWriter().write()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release() # cv2.VideoCapture().release()
cv2.destroyAllWindows() # cv2.VideoCapture().destroyAllWindows()