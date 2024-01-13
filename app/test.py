import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
import matplotlib.pyplot as plt
import os 
import numpy as np
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 3)
model_ft = model_ft.to(device)

model_path = '../model/model_ft_gpu.pth'
#Load model trained on GPU on CPU
model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '../notebook/potato'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

labels = []
datas = []
for i, (data, label) in enumerate(dataloaders['val']):
    labels.append(label)
    datas.append(data)
len(datas)

model_ft.eval()
# x, y = image_datasets['val'][0][0], image_datasets['val'][0][1]
passed = 0
for i in tqdm(range(len(datas))):
    x, y = datas[i], labels[i]
    with torch.no_grad():
        pred = model_ft(x)
        predicted, actual = class_names[pred[0].argmax(0)], class_names[y]
        # print(f'Predicted: "{predicted}", Actual: "{actual}"')
        if predicted==actual:
            passed += 1
print(f"accuracy:{passed/len(datas)}")


