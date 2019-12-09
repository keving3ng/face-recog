import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_eye_tree_eyeglasses.xml')

def focusEye(img, eyeLoc, pad=20):
    x, y, w, h = eyeLoc
    eyeCropped = img[y-pad:y+h+pad, x-pad:x+w+pad]
    cv2.imshow('Eye Cropped', eyeCropped)

def detectEyes(face):
    eyes = eye_cascade.detectMultiScale(face, 1.3, 5)
    if len(eyes) > 0:
        focusEye(face, eyes[0])

    for (x,y,w,h) in eyes:
        face = cv2.rectangle(face,(x,y),(x+w,y+h),(0,0,255),2)

def detectFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # For now, this is designed to work with one face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        detectEyes(face)

def show_webcam():
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        img = cv2.flip(img, 1)
        detectFace(img)

        cv2.imshow('Webcam Main', img)

        k = cv2.waitKey(1)
        if k == 27: # ESC key to exit
            break

    cv2.destroyAllWindows()

def main():
    show_webcam()

if __name__ == '__main__':
    main()