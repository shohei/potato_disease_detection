import streamlit as st
import torch
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import os 
import numpy as np
from tqdm import tqdm
import tempfile

tempfile_dir = tempfile.TemporaryDirectory()

st.title("Plant disease detection")
model_file = st.sidebar.file_uploader("Upload model")
image_file = st.sidebar.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 3)
model_ft = model_ft.to(device)

data_dir = '../notebook/potato'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

if model_file:
    tempfile_name = os.path.join(tempfile_dir.name, model_file.name)
    with open(tempfile_name, "wb") as f:
        f.write(model_file.getbuffer())
    model_file_path = tempfile_name
    model_ft.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))

f1 = st.empty()
f2 = st.empty()
if image_file:
    tempfile_name = os.path.join(tempfile_dir.name, image_file.name)
    with open(tempfile_name, "wb") as f:
        f.write(image_file.getbuffer())
    image_file_path = tempfile_name
    f1.image(image_file_path, caption="Uploaded image")
    model_ft.eval()

    with torch.no_grad():
        for i, (data, label) in enumerate(dataloaders['val']):

        pred = model_ft(x)
        predicted = class_names[pred[0].argmax(0)]
        # predicted, actual = class_names[pred[0].argmax(0)], class_names[y]
        # print(f'Predicted: "{predicted}", Actual: "{actual}"')
        # if predicted==actual:
        #     passed += 1
    x = 
    pred = model_ft(x)
        predicted, actual = class_names[pred[0].argmax(0)], class_names[y]

