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
from mtcnn.mtcnn import MTCNN 

image_path = "/home/kgeng/code/face-recog/data/glasses/train/glasses/TG0000.png"

detector = MTCNN()

img = cv2.imread(image_path)
result = detector.detect_faces(img)

print(result)
keypoints = result[0]['keypoints']
cv2.circle(img,(keypoints['left_eye']), 2, (0,155,255), 2)
cv2.circle(img,(keypoints['right_eye']), 2, (0,155,255), 2)
cv2.namedWindow("image")
cv2.imshow("image",img)
cv2.waitKey(0)