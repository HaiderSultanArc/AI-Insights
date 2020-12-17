import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_default.xml')

pic_number = 0
while True:
    ret, frame = cap.read()

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi = frame[y-100:y+h+100, x-100:x+w+100]

        pic_number += 1
        filename = "Image_" + str(pic_number) + ".png"
        cv2.imwrite(filename, roi)

        color = (255, 0, 0)
        stroke = 2
        cv2.rectangle(frame, (x,y), (x + w, y + h), color, stroke)

    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()