import numpy as np
import cv2
from matplotlib import pyplot as plt

face_withMask = cv2.CascadeClassifier('mask7.xml')
face_withoutMask = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascade_mouth.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (0, 255, 0)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing maskk"
not_weared_mask = "Please wear mask"
not_face_detect = "Face Does not Detect"

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facesMask = face_withMask.detectMultiScale(gray, 1.3, 5)
    faceNoMask = face_withoutMask.detectMultiScale(gray, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    if (len(facesMask) == 0 and len(mouth) != 0):
        cv2.putText(frame, not_weared_mask, org, font, font_scale, not_weared_mask_font_color, thickness, cv2.LINE_AA)
        for (x, y, w, h) in faceNoMask:
            cv2.rectangle(frame, (x, y), (x + w, y + h), not_weared_mask_font_color, 2)
    elif (len(facesMask) != 0 and len(mouth) == 0):
        cv2.putText(frame, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
        for (x, y, w, h) in facesMask:
            cv2.rectangle(frame, (x, y), (x + w, y + h), weared_mask_font_color, 2)
    else:
        cv2.putText(frame, not_face_detect, org, font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

cap.release()
cv2.destroyAllWindows()
