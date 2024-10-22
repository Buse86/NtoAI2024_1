import pandas as pd
import shutil
import os

def InTrain():
    for i in range(28015):
        shutil.move(f'/train2/{df["image_name"].iloc[i]}', f'C:/Users/Geo/PycharmProjects/nto2024AI_1/train2/{a[df["class_id"].iloc[i]]}')
    return None

def InTest():
    for i in os.listdir('train'):
        x = os.listdir(f'train/{i}')
        for j in range(len(x) // 10):
            shutil.move(f'train/{i}/{x[j]}', f'test/{i}/{x[j]}')
    return None

df = pd.read_csv('train.csv')
df = df.drop(columns=['unified_class'])

a = ['Заяц', 'Кабан', 'Кошки',  'Куньи', 'Медведь', 'Оленевые', 'Пантеры', 'Полорогие', 'Собачие', 'Сурок']

# for i in range(28015):
#     shutil.move(f'/train2/{df["image_name"].iloc[i]}', f'C:/Users/Geo/PycharmProjects/nto2024AI_1/train2/{a[df["class_id"].iloc[i]]}')

for i in os.listdir('train'):
    x = os.listdir(f'train/{i}')
    for j in range(len(x) // 10):
        shutil.move(f'train/{i}/{x[j]}', f'test/{i}/{x[j]}')