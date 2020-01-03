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

def focusEye(path, faceScale, faceNB, eyeScale, eyeNB):
    eyeFound = 0
    faceFound = 0
    
    img = cv2.imread(path)
    # Convert img to grayscale and detect face using haar'
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, faceScale, faceNB)

    if len(faces) > 0:
        faceFound = 1
    
        # Store the location of the face and crop the image according to the dimensions and location
        fx, fy, fw, fh = faces[0]
        face = img[fy:fy+fh, fx:fx+fw]

        # Detect eyes on the face
        eyes = eye_cascade.detectMultiScale(face, eyeScale, eyeNB)

        # Raise exception if no eyes found
        if len(eyes) > 0:
            eyeFound = 1
        
    return faceFound, eyeFound

def find_weights():
    path = "/home/kgeng/code/face-recog/data/glasses/train/glasses"
    facesFound = 0
    eyesFound = 0
    best = {'eyes': 0, 'fs':0, 'fn':0, 'es':0, 'en':0}
    scales = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]
    nbs = [3, 4, 5, 6]

    for faceScale in scales:
        for faceNB in nbs:
            for eyeScale in scales:
                for eyeNB in nbs:
                    for f in os.listdir(path):
                        face, eye = focusEye(os.path.join(path,f), faceScale, faceNB, eyeScale, eyeNB)
                        facesFound += face
                        eyesFound += eye
                    fPerc = facesFound / len(os.listdir(path)) * 100
                    ePerc = eyesFound / len(os.listdir(path)) * 100
                    # print("face: {}, {} - eye: {}, {}, acc: face={}%, eye={}%".format(faceScale, faceNB, eyeScale, eyeNB, fPerc, ePerc))
                    if eyesFound > best.get('eyes'):
                        best['eyes'] = eyesFound
                        best['fs'] = faceScale
                        best['fn'] = faceNB
                        best['es'] = eyeScale
                        best['en'] = eyeNB
                    if eyesFound == best.get('eyes'):
                        print("EQUAL - face: {}, {} - eye: {}, {}, acc: face={}%, eye={}%".format(faceScale, faceNB, eyeScale, eyeNB, fPerc, ePerc))
                    
                    facesFound = 0
                    eyesFound = 0
    print(best)
if __name__ == "__main__":
    find_weights()