import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye_tree_eyeglasses.xml')

def focusEye(img, eyeLoc):
    x, y, w, h = eyeLoc
    eyeCropped = img[y:y+h, x:x+w]
    cv2.imshow('Eye Cropped', eyeCropped)

def detectEyes(img):
    eyes = eye_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in eyes:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    if len(eyes) > 0:
        focusEye(img, eyes[0])

def captureAndDetect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = measureFace(gray)
    detectEyes(gray)
    cv2.imshow('Eye Detection', img)

def measureFace(img):
    x, y, w, h = face_cascade.detectMultiScale(img, 1.3, 5)
    return w, h

def show_webcam():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)
        cv2.imshow('Webcam Main', img)
        k = cv2.waitKey(1)

        if k == 32: # Space to Capture and Detect
            captureAndDetect(img)
        elif k == 27: # ESC key to exit
            break
        elif k == -1:
            continue

    cv2.destroyAllWindows()

def main():
    cv2.namedWindow('Webcam Main')
    cv2.namedWindow('Eye Detection')
    show_webcam()

if __name__ == '__main__':
    main()