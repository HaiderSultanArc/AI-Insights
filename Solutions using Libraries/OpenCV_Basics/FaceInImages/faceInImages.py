import cv2, os
import  numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')

current_id = 0
label_ids = {}

for root, dirs, files in os.walk(image_dir):

    filenumber = 0
    for file in files:
        print(filenumber)
        if file.endswith("png") or file.endswith("jpg"):

            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]

            gray = cv2.imread(path)
            # gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                img_item = str(id_) + "_Image_" + str(filenumber) + ".jpg"
                print(img_item)
                cv2.imwrite(img_item, roi)

            filenumber += 1