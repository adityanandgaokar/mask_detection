
import cv2

import numpy as np

detector = cv2.CascadeClassifier('C:/Users/Aditya/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)

count = 0

while(True):

    ret, img = capture.read()

    faces = detector.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:

        rec = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

        count += 1

        number = count
        
        with_mask = 'D:/Projects/Mask_detection/dataset/with_mask/'

        without_mask = 'D:/Projects/Mask_detection/dataset/without_mask/'

        cv2.imwrite(with_mask + 'aditya' + str(number) + '.jpg', img[y:y+h, x:x+w])

        cv2.imshow('frame', rec)

    if count == 500:
        break

##    if cv2.waitKey(1) & 0xFF == ord('q'):
##        
##        break

capture.release()
cv2.destroyAllWindows()
