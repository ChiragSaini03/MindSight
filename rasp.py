import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import torch

import pickle

#Adjust alpha values to suit your need
alpha = 0.2
previous_depth = 0.0

#Applying exponential moving average filter
def apply_ema_filter(current_depth):
    global previous_depth
    filtered_depth = alpha * current_depth + (1 - alpha) * previous_depth
    previous_depth = filtered_depth  # Update the previous depth value
    return filtered_depth

def computeDist(objectresult, depth_map, d1, d2):

    all_dists = []
    dist_from_obj = {}
    dcur1 = depth_map[0][depth_map.shape[1]-1]
    dcur2 = depth_map[depth_map.shape[0]-1][0]
    
    # y = a*x + b
    a = (d1 - d2) / (dcur1 - dcur2)
    b = d1 - (a * dcur1)
    for detection in objectresult.detections:
        # Draw bounding_box

        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        mid_point = bbox.origin_x + bbox.width//2, bbox.origin_y + bbox.height//2
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        
        dist_from_obj[category_name] = {}
        # as mid_point[1] indicate the height + x coordinate of object 
        dist_from_obj[category_name]["dist"] = (depth_map[mid_point[1]][mid_point[0]]*a) +b
        dist_from_obj[category_name]["prob"] = probability

        x3 = depth_map.shape[1]
        y3 = depth_map.shape[0]
        x1 = x3//3
        y1 = y3//3
        x2 = (2*x3)//3
        y2 = (2*x3)//3

        x = mid_point[0]
        y = mid_point[1]

        if x<=x1 and y<=y1:
            dir = "Top Left"
        elif x<=x1 and y>=y1 and y<=y2:
            dir = "Left"
        elif x<=x1 and y>=y2:
            dir = "Bottom Left"
        elif x>=x1 and x<=x2 and y<=y1:
            dir = "Top"
        elif x>=x1 and x<=x2 and y>=y1 and y<=y2:
            dir = "Center"
        elif x>=x1 and x<=x2 and y>=y2:
            dir = "Bottom"
        elif x>=x2 and y<=y1:
            dir = "Top Right"
        elif x>=x2 and y>=y1 and y<=y2:
            dir = "Right"
        else:
            dir = "Bottom Right"

        dist_from_obj[category_name]["dir"] = dir

        all_dists.append(dist_from_obj)
        dist_from_obj = {}

    return all_dists;

def dist_to_response(dist_from_object): 
    with open("pastrec.dat","rb") as f:
        prev = pickle.load(f)
    # analysing a change in past environment and current environment and then reporting it to the user
    # data = {
    #     "obj1": {
    #         "dist": 100,
    #         "dir": "left"
    #     }, 
    #     "obj2": {
    #         "dist": 200,
    #         "dir": "top-left"
    #     },
    # }
    data = dist_from_object
    ## processing code Pending to process and compute the data dictionary

    final_data = {}
    for i in data.keys():
        for j in prev.keys():
            if (i==j):
                if (data[i]["dir"]!=prev[j]["dir"]):
                    final_data[i] = data[i]
                elif (abs(data[i]["dist"] - prev[j]["dist"]) >= 150):
                    final_data[i] = data[i];
        if i not in list(prev.keys()):
            final_data[i]=data[i]

    with open("pastrec.dat","wb") as f:
        pickle.dump(final_data,f)

    res = ""
    for i in final_data.keys():
        res = res + f"{i} is at {final_data[i]['dir']} {final_data[i]['dist']} away "
    return res

# unused due to extreme complexity - O(480x640)
def normalize_depth_map(depth_map, d1, d2):
    dcur1 = depth_map[0][depth_map.shape[1]-1]
    dcur2 = depth_map[depth_map.shape[0]-1][0]

    # y = a*x + b
    a = (d1 - d2) / (dcur1 - dcur2)
    b = d1 - (a * dcur1)

    # Normalize whole depth map
    for i in range(0,depth_map.shape[0]):
        for j in range(0,depth_map.shape[1]):
            depth_map[i][j] = (a * depth_map[i][j]) + b

    return depth_map

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
    # print(detection_result)

    input_batch = transform(frm).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        # print(prediction)

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

        d1 = 400  # distance calculated from first ultrasonic sensor
        d2 = 70   # distance calculated from second ultrasonic sensor

        # currently d1 is top left and d2 is bottom right
        # depth_map = normalize_depth_map(depth_map,d1,d2)


        print(depth_map[10][10])
        print(depth_map[0][0],"\t",depth_map[0][depth_map.shape[1]-1])
        print("\t",depth_map[int(depth_map.shape[0]/2)][int(depth_map.shape[1]/2)],"\t")
        print(depth_map[depth_map.shape[0]-1][0],"\t",depth_map[depth_map.shape[0]-1][depth_map.shape[1]-1])

        dist_from_obj = computeDist(detection_result, depth_map, d1, d2)
        print(dist_from_obj)

        # # Converting the distances of object to respective text--
        # # This should be executed parallely while the next iteration can be runned

        # dist_to_response(dist_from_obj)  # this should be async function

        # depth_map = cv2.normalize(depth_map,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
 
        # depth_map = (depth_map*255).astype(np.uint8)
        # depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        # cv2.imshow('Depth Map', depth_map)
        # if (cv2.waitKey(5) & 0xFF == 27):
        #     break;

cam.release()


'''
https://medium.com/artificialis/getting-started-with-depth-estimation-using-midas-and-python-d0119bfe1159

https://medium.com/artificialis/swift-and-simple-calculate-object-distance-with-ease-in-just-few-lines-of-code-38889575bb12

https://github.com/isl-org/MiDaS/issues/42
'''