#!env/bin/python
"""
This class is used to detect an eye from either a continuous video input, or a still image. 
It uses OpenCV and haarcascades to identify a face and then find an eye on the face. 

Usage:
    Webcam:
        $ crop_eye.py

    Image:
        $ crop_eye.py -i /path/to/image

To exit, press 'ESC'
"""

import argparse
import os
import datetime

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_eye_tree_eyeglasses.xml')

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', default=None, help='Image to process. If no image is given, then webcam will start.')
parser.add_argument('--loglevel', '-l', default=1, help='LOG_LEVEL Level. 0 - SILENT, 1 - SIG EVENTS, 2 - ALL')

timestamp = lambda : "[{}]".format(str(datetime.datetime.now()).split('.')[0])

class DetectGlasses:
    # LOG_LEVEL LEVELS:
    #   0 - SILENT
    #   1 - SIG EVENTS
    #   2 - ALL
    LOG_LEVEL = 1

    def __init__(self, arg_img, LOG_LEVEL):
        LOG_LEVEL = LOG_LEVEL

        if arg_img == None:
            self.show_webcam()
        elif not os.path.exists(arg_img):
            raise Exception("Path \"{}\" does not exist.".format(arg_img))
        elif os.path.exists(arg_img):
            img = cv2.imread(arg_img)
            cv2.imshow('Main', img)
            self.detectFace(img)
            while True:
                if cv2.waitKey(1) == 27: # ESC key to exit
                    break
            cv2.destroyAllWindows()
        else:
            exit()

    def focusEye(self, img, eyeLoc, pad):
        '''
            Crops location of eye with padding to capture area with eye glasses

            Args:
                img                   (img): image of face
                eyeLoc (int, int, int, int): x, y coords of eye, width, height of eye
                pad              (int, int): padding around eye to capture glasses
            Returns:  None
            Raises:   None
        '''
        if self.LOG_LEVEL > 1: print("{} Eyes Detected".format(timestamp()))
        x, y, w, h = eyeLoc
        wPad, hPadBot, hPadTop = pad

        eyeCropped = img[y-hPadTop:y+h+hPadBot, x-wPad:x+w+wPad]
        cv2.imshow('Eye Cropped', eyeCropped)

    def detectEyes(self, face, pad):
        '''
            Detects eyes on image of face, then draws a rectangle on the first one detected.
            Then, calls another function to crop out the area around the eye.

            Args:
                face     (img): image of the face
                pad (int, int): padding of eye area
            Returns:  None
            Raises:   None
        '''
        if self.LOG_LEVEL > 1: print("{} Detecting Eyes".format(timestamp()))
        eyes = eye_cascade.detectMultiScale(face, 1.3, 5)
        if len(eyes) > 0:
            x, y, w, h = eyes[0]
            face = cv2.rectangle(face,(x,y),(x+w,y+h),(0,0,255),2)

            self.focusEye(face, eyes[0], pad)

    def calcPadding(self, w, h, wPct=0.45, hPct=0.35, split=0.7):
        '''
            Calculates padding around eye region based on face area size to reduce search area for eyeglasses.
            Since glasses usually will typically down below the eye instead of above, the height padding is split
            unevenly over the top and bottom

            Args:
                w       (int): width of face
                h       (int): height of face
                wPct  (float): ratio to scale the width padding. Approximate percentage of glasses coverage of face of ONE eye
                hPct  (float): see above, but for height on face.
                split (float): split of top/bottom padding of eye region
            Returns:
                'face width padding', 'face height top padding', 'face height bottom padding'
            Raises:  None
        '''
        if self.LOG_LEVEL > 1: print("{} Calculating padding".format(timestamp()))
        return int(w * wPct / 4), int(h * hPct / 2 * split), int(h * hPct / 2 * (1 - split))

    def detectFace(self, img):
        '''
            Detects face(s) using haar cascade from grayscale image. When face is found, goes to detect eyes
            on image of face area (reducing search area).

            Args:
                img (img): image from webcam input to process
            Returns:  None
            Raises:   None
        '''
        if self.LOG_LEVEL > 1: print("{} Detecting face".format(timestamp()))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # For now, this is designed to work with one face
        if len(faces) > 0:
            x, y, w, h = faces[0]
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255))
            cv2.putText(img, "Face: {}x{}".format(w, h), (x, y+h+15), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)
            cv2.imshow('Main', img)
            face = img[y:y+h, x:x+w]
            self.detectEyes(face, self.calcPadding(w, h))

    def show_webcam(self):
        '''
            This function loops to call functions to show webcam video and process the images.

            Args:     None
            Returns:  None
            Raises:   None
        '''
        if self.LOG_LEVEL > 0:
            print("{} Starting webcam".format(timestamp()))
        cam = cv2.VideoCapture(0)
        ret_val = True
        while ret_val:
            ret_val, img = cam.read()
            img = cv2.flip(img, 1)
            self.detectFace(img)

            cv2.imshow('Main', img)

            k = cv2.waitKey(1)
            if k == 27: # ESC key to exit
                break

        cv2.destroyAllWindows()
        print("{} Webcam closed".format(timestamp()))

if __name__ == '__main__':
    args = parser.parse_args()
    DetectGlasses(args.image, args.loglevel)