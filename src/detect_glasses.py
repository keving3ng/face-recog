import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_eye_tree_eyeglasses.xml')

def focusEye(img, eyeLoc, pad):
    x, y, w, h = eyeLoc
    wPad, hPad = pad

    eyeCropped = img[y-hPad:y+h+hPad, x-wPad:x+w+wPad]
    cv2.imshow('Eye Cropped', eyeCropped)

def detectEyes(face, pad):
    eyes = eye_cascade.detectMultiScale(face, 1.3, 5)
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        face = cv2.rectangle(face,(x,y),(x+w,y+h),(0,0,255),2)

        focusEye(face, eyes[0], pad)

def calcPadding(w, h, wPct=0.5, hPct=0.33):
    nw, nh = int(w * wPct / 4), int(h * hPct / 4)
    print("w={} -> {}, h={} -> {}".format(w, nw, h, nh))
    return nw, nh

def detectFace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # For now, this is designed to work with one face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = img[y:y+h, x:x+w]
        detectEyes(face, calcPadding(w, h))

def show_webcam():
    '''
        This function loops to call functions to show webcam video and process the images.

        Args:     None
        Returns:  None
        Raises:   None
    '''
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