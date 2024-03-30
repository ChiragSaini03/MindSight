import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch

#Adjust alpha values to suit your need
alpha = 0.2
previous_depth = 0.0
#Applying exponential moving average filter
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth

def computeDist(objectresutl, depth_map):
    pass

def dist_to_response(dist_from_object):  # this to be async function
    pass

# Get Camera Feed and pass it on to the server
# Which will be a POST request
cam = cv2.VideoCapture(0)

# Object Recognition Model Loading
base_options = python.BaseOptions(model_asset_path='main/object_recog/efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Depth Estimation Model Loading
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDas",model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDas", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else :
    transform = midas_transforms.small_transform


while cam.isOpened() :

    _, frm = cam.read()

    np_image = np.array(frm)
    print(np_image.shape)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frm)
    detection_result = detector.detect(image)
    print(detection_result)

    input_batch = transform(frm).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        print(prediction)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frm.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # alpha ema filter
        # depth_map = (depth_map*alpha + previous_depth*(1-alpha))
        # previos_depth = depth_map

        # print(depth_map)
        print(depth_map[10][10])
        print(depth_map[0][0],"\t",depth_map[0][depth_map.shape[1]-1])
        print("\t",depth_map[int(depth_map.shape[0]/2)][int(depth_map.shape[1]/2)],"\t")
        print(depth_map[depth_map.shape[0]-1][0],"\t",depth_map[depth_map.shape[0]-1][depth_map.shape[1]-1])

        depth_map = cv2.normalize(depth_map,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)

        depth_map = (depth_map*255).astype(np.uint8)
        depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        dist_from_obj = computeDist(detection_result, depth_map)

        # Converting the distances of object to respective text--
        # This should be executed parallely while the next iteration can be runned

        dist_to_response(dist_from_obj)  # this should be async function

 
        cv2.imshow('Depth Map', depth_map)
        if (cv2.waitKey(5) & 0xFF == 27):
            break;

cam.release()


'''
https://medium.com/artificialis/getting-started-with-depth-estimation-using-midas-and-python-d0119bfe1159

https://medium.com/artificialis/swift-and-simple-calculate-object-distance-with-ease-in-just-few-lines-of-code-38889575bb12
'''