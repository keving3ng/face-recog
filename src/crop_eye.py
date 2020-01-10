#!env/bin/python
"""
    Result Example:
        [
            {
                'box': [32, 25, 71, 95],
                'confidence': 0.9514315128326416, 
                'keypoints': {
                    'left_eye': (42, 60), 
                    'right_eye': (77, 60), 
                    'nose': (56, 85), 
                    'mouth_left': (46, 100), 
                    'mouth_right': (74, 100)
                }
            }
        ]
"""

import cv2
import os
from mtcnn.mtcnn import MTCNN 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def calcPadding(w, h, wPct=0.75, hPct=0.80, split=0.55):
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

def find_eye(img):
    detector = MTCNN()
    result = detector.detect_faces(img)
    
    if result:
        keypoints = result[0]['keypoints']
        bounding_box = result[0]['box']
        eye = keypoints['left_eye']
        wPad, hPadBot, hPadTop = calcPadding(bounding_box[3], bounding_box[2])

        # Crop the eye and return it
        eyeCropped = img[eye[1] - hPadTop:eye[1] + hPadBot, eye[0] - wPad:eye[0] + wPad]
        return eyeCropped # Returns as np array

if __name__ == "__main__":
    img = cv2.imread("./data/glasses/train/glasses/TG0024.png")
    find_eye(img)