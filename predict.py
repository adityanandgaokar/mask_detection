import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2



protopath = os.path.sep.join(['D:/Projects/Mask_detection/face_detector', 'deploy'])

weightpath = os.path.sep.join(['D:/Projects/Mask_detection/face_detector', 'res10_300x300_ssd_iter_140000.caffemodel'])


face_detec = cv2.dnn.readNet(protopath, weightpath)

    
trained_model = load_model('D:/Projects/Mask_detection/Models/mask_detection.h5')


video = VideoStream(src=0).start()
time.sleep(2.0)

def detection_and_prediction(frame, weightpath, protopath):

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (400,400),
                                 (104.0, 177.0, 123.0))

    face_detec.setInput(blob)

    detection = face_detec.forward()


    roi = []
    location = []
    prediction = []

    for i in range(0, detection.shape[2]):
    
        conf = detection[0,0,i,2]


        if conf > 0.5:

            box = detection[0,0,i,3:7] * np.array([W,H,W,H])

            (x_start, y_start, x_end, y_end) = box.astype('int')

            (x_start, y_start) = (max(0, x_start), max(0, y_start))
            (x_end, y_end) = (min(W-1, x_end), min(H-1, y_end))

            face = frame[y_start:y_end, x_start:x_end]

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            face = img_to_array(face)
            face = preprocess_input(face)

            roi.append(face)
            location.append((x_start, y_start, x_end, y_end))

    if len(roi) > 0:

        roi = np.array(roi, dtype='float32')
        prediction = trained_model.predict(roi, batch_size = 32)

                
            
    return (location, prediction)            

while True:

    frame = video.read()

    frame = imutils.resize(frame, width = 400)

    (location, prediction) = detection_and_prediction(frame, weightpath, protopath)

    for (box, pred) in zip(location, prediction):

        (x_start, y_start, x_end, y_end) = box
        (mask, withoutmask) = pred

        if mask > withoutmask:
            label = 'Mask'
            color = (0,255,0)

        else :
            label = 'No Mask'
            color = (0,0,255)


        label = '{}: {:.2f}%'.format(label, max(mask, withoutmask) * 100)

        cv2.putText(frame, label, (x_start, y_start - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)

        cv2.imshow('frame', frame)
        


        


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cv2.destroyAllWindows()
video.stop()
