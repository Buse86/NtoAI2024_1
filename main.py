import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm
import random
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from PIL import Image
import pandas as pd

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# data_count = [915, 1964, 930, 747, 2394, 11019, 4282, 1882, 1766, 720]
# data_weight = [d/len(data_count) for d in data_count]
# data_weight_tensor = torch.FloatTensor(data_weight).cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = models.resnet34(pretrained=True)

# Заменим последний слой (fully connected) так, чтобы количество выходных каналов соответствовало 3 классам
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

train_data_dir = 'train2'
val_data_dir = 'test22'

# Определим трансформации
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Создадим датасеты
train_dataset = ImageFolder(train_data_dir, transform=train_transforms)
val_dataset = ImageFolder(val_data_dir, transform=val_transforms)

# Создадим датагенераторы
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

classes_list = train_dataset.classes

# Определим функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# число эпох
num_epochs = 50

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_accuracy = 0
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Валидация модели
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Сохранение лучшей модели на основе валидационной точности
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model1.pth')
        print('Saved best model!')

    # Сохранение последней актуальной модели
    torch.save(model.state_dict(), 'last_model.pth')
    print()

print('Training and validation complete!')

f = False

if f:
    ans = pd.DataFrame(columns=['image_name', 'predicted_class'])

    # Загрузим предварительно обученную модель ResNet18
    model = models.resnet18()
    # Заменим последний слой (fully connected) так, чтобы количество выходных каналов соответствовало 3 классам
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Загрузка весов модели
    model.load_state_dict(torch.load('last_model.pth'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

            # class_name = class_names[top_class]
            print(top_class)
            ans.loc[idx] = [file, top_class]
            idx += 1


    ans.to_csv('answer3.csv', index=False)
    print('sucsses')