print('importing library')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import os 
from torch.utils.data import Dataset
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pdb

model_file_path = "../model/model_ft_gpu.pth"
model_ft = models.resnet18(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 3)
model_ft = model_ft.to(device)
model_ft.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
model_ft.eval()
print('model loaded.')

preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                  ])

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

cap = cv2.VideoCapture('test.mov')

font                   = cv2.FONT_HERSHEY_SIMPLEX
topLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,255,255)
thickness              = 1
lineType               = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', frame) 

    pil_image = Image.fromarray(frame)
    img_tensor = preprocess(pil_image)
    img_tensor = img_tensor.unsqueeze(0)
    cropped_img = img_tensor.squeeze().permute(1, 2, 0).numpy()
    cv2.imshow('frame2',cropped_img)

    with torch.no_grad():
        pred = model_ft(img_tensor)
        predicted = class_names[pred[0].argmax(0)]
        pred_index = pred[0].argmax(0)
        if(pred_index)!=0:
            fontColor = (255,0,0)
        if(pred_index)!=1:
            fontColor = (128,128,128)
        if(pred_index)!=2:
            fontColor = (255,255,255)

        blank_image = np.zeros((50,400,3), np.uint8)
        cv2.putText(
            img = blank_image, 
            text = predicted,
            org = topLeftCornerOfText,
            fontFace = font,
            fontScale = fontScale,
            color = fontColor,
            thickness = thickness,
        )
        print(f'Predicted: "{predicted}"')
        cv2.imshow('frame3',blank_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
