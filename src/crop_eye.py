#!env/bin/python
import argparse
import os
import datetime

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/cascades/haarcascade_eye_tree_eyeglasses.xml')

def calcPadding(w, h, wPct=0.45, hPct=0.35, split=0.7):
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
    return int(w * wPct / 4), int(h * hPct / 2 * split), int(h * hPct / 2 * (1 - split))

def focusEye(img):
    # Convert img to grayscale and detect face using haar'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Raise exception if no faces are found
    if len(faces) < 1:
        raise Exception("No faces found")
    
    # Store the location of the face and crop the image according to the dimensions and location
    fx, fy, fw, fh = faces[0]
    face = img[fy:fy+fh, fx:fx+fw]

    # Detect eyes on the face
    eyes = eye_cascade.detectMultiScale(face, 1.3, 5)

    # Raise exception if no eyes found
    if len(eyes) < 1:
        raise Exception("No eyes found")
    
    # Store the location and dimension of one (1) eye
    ex, ey, ew, eh = eyes[0]

    # Calculate the crop padding around the eye for where eyeglasses would be
    wPad, hPadBot, hPadTop = calcPadding(ew, eh)
    
    # Crop the eye and return it
    eyeCropped = img[ey-hPadTop:ey+eh+hPadBot, ex-wPad:ex+ew+wPad]
    print("{}w x {}h".format(ew, eh))
    return eyeCropped