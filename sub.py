import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd

from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from PIL import Image

ans = pd.DataFrame(columns=['image_name', 'predicted_class'])

# Загрузим предварительно обученную модель ResNet18
model = models.resnet34(pretrained=True)
# Заменим последний слой (fully connected) так, чтобы количество выходных каналов соответствовало 3 классам
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Загрузка весов модели
model.load_state_dict(torch.load('best_model1.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
model.eval()

class_names = ['Заяц', 'Кабан', 'Кошки',  'Куньи', 'Медведь', 'Оленевые', 'Пантеры', 'Полорогие', 'Собачие', 'Сурок']

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

idx = 0

for file in os.listdir('test'):
    image_path = f'test/{file}'
    image = Image.open(image_path)
    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu()

        top_prob, top_class = torch.topk(probabilities, 1)
        top_prob = top_prob.item()
        top_class = top_class.item()
        class_name = class_names[top_class]
        print(class_name, top_class, top_prob)

        ans.loc[idx] = [file, top_class]
        idx += 1

ans.to_csv('answer4.csv', index=False)
print('sucsses')