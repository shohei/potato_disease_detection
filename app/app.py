import streamlit as st
import torch
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import os 
from tqdm import tqdm
import tempfile
from torch.utils.data import Dataset

tempfile_dir = tempfile.TemporaryDirectory()

st.title("Plant disease detection")
# model_file = st.sidebar.file_uploader("Upload model")
# model_file = st.sidebar.selectbox("Select model", ("model_ft.pth","a" )) # セレクトボックス
model_file = st.sidebar.radio("Select model", ("model_ft_gpu.pth","")) 
image_file = st.sidebar.file_uploader("Upload image", type=['png', 'jpg', 'jpeg'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 3)
model_ft = model_ft.to(device)

data_dir = '../notebook/potato'

if model_file:
    model_file_path = os.path.join(os.getcwd(),"../model",model_file)
    model_ft.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))

f1 = st.empty()
f2 = st.empty()
if image_file:
    image_folder_path_root = os.path.join(tempfile_dir.name, 'image') 
    image_folder_path_sub = os.path.join(image_folder_path_root, 'no_label') 
    os.mkdir(image_folder_path_root)
    os.mkdir(image_folder_path_sub)
    tempfile_name = os.path.join(tempfile_dir.name, 'image', 'no_label', image_file.name)
    with open(tempfile_name, "wb") as f:
        f.write(image_file.getbuffer())
    image_file_path = tempfile_name
    f1.image(image_file_path, caption="Uploaded image")
    model_ft.eval()

    data_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                      ])
    image_dataset = datasets.ImageFolder(
                      image_folder_path_root,
                      data_transforms
                    )
    dataloaders = torch.utils.data.DataLoader(
                    image_dataset,
                    batch_size=1, shuffle=True, num_workers=0)

    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

    with torch.no_grad():
        x = dataloaders.dataset[0][0].unsqueeze(0)
        pred = model_ft(x)
        predicted = class_names[pred[0].argmax(0)]
        f2.write(f'Predicted: "{predicted}"')