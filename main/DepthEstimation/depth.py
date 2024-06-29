import cv2
import torch
import time
import numpy as np

# Load a MiDas model for depth estimation
# model_type = "DPT_Large"  # MiDas v3 - Large (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid" # MiDas v3 - Hybrid (medium accuracy, medium inference speed)
model_type = "MiDaS_small" # MiDas v2.1 - Smalll (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDas",model_type)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

# Load transforms to resize and normalize the image
midas_transforms = torch.hub.load("intel-isl/MiDas", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else :
    transform = midas_transforms.small_transform

cap = cv2.VideoCapture(0)

def normalize_depth_map(depth_map, d1, d2):
    dcur1 = depth_map[0][depth_map.shape[1]-1]
    dcur2 = depth_map[depth_map.shape[0]-1][0]
    

    # y = a*x + b
    a = (d1 - d2) / (dcur1-dcur2)
    b = d1 - (a * dcur1)

    # Normalize whole depth map
    for i in range(0,depth_map.shape[0]):
        for j in range(0,depth_map.shape[1]):
            depth_map[i][j] = (a * depth_map[i][j]) + b

    return depth_map


while cap.isOpened():
    
    success, img = cap.read()

    start = time.time()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        d1 = 400
        d2 = 70

        print(depth_map[0][depth_map.shape[1]-1])
        print(depth_map[depth_map.shape[0]-1][0])

        # depth_map = normalize_depth_map(depth_map, d1, d2)

        # # print(depth_map.shape)

        # print(depth_map)
        print(depth_map[0][0],"\t",depth_map[0][depth_map.shape[1]-1])
        print("\t",depth_map[int(depth_map.shape[0]/2)][int(depth_map.shape[1]/2)],"\t")
        print(depth_map[depth_map.shape[0]-1][0],"\t",depth_map[depth_map.shape[0]-1][depth_map.shape[1]-1])

        # time.sleep(2)
        depth_map = cv2.normalize(depth_map,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)

        end = time.time()
        totalTime = end-start

        fps = 1/totalTime

        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        depth_map = (depth_map*255).astype(np.uint8)
        # depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('Image', img)
        cv2.imshow('Depth Map', depth_map)

        if (cv2.waitKey(5) & 0xFF == 27):
            break;

cap.release()

