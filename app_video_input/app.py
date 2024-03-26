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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow('frame', frame) 

    pil_image = Image.fromarray(frame)
    img_tensor = preprocess(pil_image)
    img_tensor = img_tensor.unsqueeze(0)
    plt.imshow(img_tensor.squeeze().permute(1, 2, 0))

    with torch.no_grad():
        pred = model_ft(img_tensor)
        predicted = class_names[pred[0].argmax(0)]
        plt.text(-10, -10, predicted, horizontalalignment='center', verticalalignment='center')
        if pred[0].argmax(0)!=2:
            print(f'Predicted: "{predicted}"')
            plt.text(-10, -20, "detected", color="red", horizontalalignment='center', verticalalignment='center')
            plt.pause(5)
    plt.draw()
    plt.pause(0.000001)
    plt.cla()

cap.release()
cv2.destroyAllWindows()
