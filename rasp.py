import cv2
import numpy as np

# Get Camera Feed and pass it on to the server
# Which will be a POST request
cam = cv2.VideoCapture(0)

while (True) :

    _, frm = cam.read()

    np_image = np.array(frm)
    print(np_image.shape)

    
    

cam.release()
