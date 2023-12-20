# Facial Recognition Software
# Author: Hamza Niaz
# References: https://realpython.com/face-recognition-with-python/
'''
LIMITATIONS
- Will not detect faces that are not upright
- Needs good lighting, darker faces are harder to detect
- Works most of the time, but not always
'''

import cv2
import os

# get user supplied values
imageFile = input("Enter image file: ")
cascPath = "haarcascade_frontalface_default.xml"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
imagePath = os.path.join(__location__, imageFile)

# create haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)

# read image, convert to grayscale
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in image
faces = faceCascade.detectMultiScale(
    gray, 
    scaleFactor = 1.2, 
    minNeighbors = 3, 
    minSize = (30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

# print message
print("Found {0} faces!".format(len(faces)))

# draw rectangle around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)