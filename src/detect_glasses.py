import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_eye_tree_eyeglasses.xml')

def focusEye(img, eyeLoc, pad):
    '''
        Crops location of eye with padding to capture area with eye glasses

        Args:
            img                   (img): image of face
            eyeLoc (int, int, int, int): x, y coords of eye, width, height of eye
            pad              (int, int): padding around eye to capture glasses
        Returns:  None
        Raises:   None
    '''
    x, y, w, h = eyeLoc
    wPad, hPad = pad

    eyeCropped = img[y-hPad:y+h+hPad, x-wPad:x+w+wPad]
    cv2.imshow('Eye Cropped', eyeCropped)

def detectEyes(face, pad):
    '''
        Detects eyes on image of face, then draws a rectangle on the first one detected.
        Then, calls another function to crop out the area around the eye.

        Args:
            face     (img): image of the face
            pad (int, int): padding of eye area
        Returns:  None
        Raises:   None
    '''
    eyes = eye_cascade.detectMultiScale(face, 1.3, 5)
    if len(eyes) > 0:
        x, y, w, h = eyes[0]
        face = cv2.rectangle(face,(x,y),(x+w,y+h),(0,0,255),2)

        focusEye(face, eyes[0], pad)

def calcPadding(w, h, wPct=0.5, hPct=0.33):
    '''
        Calculates padding around eye region based on face area size to reduce search area for eyeglasses.

        Args:
            w      (int): width of face
            h      (int): height of face
            wPct (float): Ratio to scale the width padding. Approximate percentage of glasses coverage of face of ONE eye
            hPct (float): See above, but for height on face.
        Returns:
            'face width padding', 'face height padding'
        Raises:  None
    '''
    return int(w * wPct / 4), int(h * hPct / 4)

def detectFace(img):
    '''
        Detects face(s) using haar cascade from grayscale image. When face is found, goes to detect eyes
        on image of face area (reducing search area).

        Args:
            img (img): image from webcam input to process
        Returns:  None
        Raises:   None
    '''
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