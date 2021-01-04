import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)  # VideoCapture captures the video, its arguments are Camera Index number (Simply the number of camera to be used, out of all connected ones) or name of Video file

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()  # cv2.VideoCapture().read(),  read() method reads the Video we are Capturing, it returns True in case of Correctly capturing of Video and False otherwise

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        print(x, y, w, h)

        roi_gray = gray[y:y+h, x:x+w] # (y-cord_start, y-cord_end) , (x-cord_start, x-cord_end)
        roi_color = frame[y:y+h, x:x+w] # (y-cord_start, y-cord_end) , (x-cord_start, x-cord_end)

        # RECOGNIZER, we can use Deep Learning Models here

        img_item = "my-img.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0) # BGR not RGB, no one knows why....
        stroke = 2 # thickness of rectangle
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke) # This method gets, The Frame or Window on which it has to draw, Starting Co-ordinates, Ending Co-ordinates, color for Rectangle, Stroke of Rectangle

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release() # cv2.VideoCapture().release()
cv2.destroyAllWindows()