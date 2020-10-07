import cv2
import numpy as np
import dlib
import imutils
from imutils import face_utils

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
palm_cascade = cv2.CascadeClassifier('palm.xml')
fist_cascade = cv2.CascadeClassifier('fist.xml')
closed_palm_cascade = cv2.CascadeClassifier('closed_frontal_palm.xml')
"""
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
"""
while(1):

    _, frame = cap.read()

    lower_blue = np.array([89,80,80])
    upper_blue = np.array([107,255,255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(frame,frame, mask= mask)

    thresh = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, hierarchy = cv2.findContours(thresh, 1, 2);
    cnt = contours[0];
    perimeter = cv2.arcLength(cnt, True);
    rect = cv2.minAreaRect(cnt);
    box = cv2.boxPoints(rect)
    box = np.int0(box);

    """
    for (x, y, w, h) in mask:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(res, (x, y), (x+w, y+h), (255, 0, 0), 2)
    """


    faces = face_cascade.detectMultiScale(hsv, 1.1, 4);
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(res, (x, y), (x+w, y+h), (255, 0, 0), 2)
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray,1)
    for(i, rects) in enumerate(rects):
        shape = predictor(gray, rects)
        shape = face_utils.shape_to_np(shape)
    for(name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        clone = frame.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
        for(x,y) in shape[i:j]:
            cv2.circle(clone, (x,y), 1, (0,0,255),-1)
        (x,y,w,h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = frame[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250,inter=cv2.INTER_CUBIC)
    """
    #output = face_utils.visualize_facial_landmarks(frame, shape)
    #cv2.imshow("Detect facial features", output);
    frame = cv2.drawContours(frame, [box], 0,(0,0,255), 2)
    mask = cv2.drawContours(mask, [box], 0,(0,0,255), 2)
    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
