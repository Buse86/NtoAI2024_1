import pandas as pd
import shutil
import os
import random
# def InTrain(df, a):
#     for i in range(28015):
#         shutil.move(f'/train2/{df["image_name"].iloc[i]}', f'C:/Users/Geo/PycharmProjects/nto2024AI_1/train2/{a[df["class_id"].iloc[i]]}')
#     return None
#
# def InTest():
#     for i in os.listdir('train2'):
#         x = os.listdir(f'train2/{i}')
#         for j in range(len(x) // 10):
#             shutil.move(f'train2/{i}/{x[j]}', f'test/{i}/{x[j]}')
#     return None

df = pd.read_csv('train.csv')
df = df.drop(columns=['unified_class'])

a = ['Заяц', 'Кабан', 'Кошки',  'Куньи', 'Медведь', 'Оленевые', 'Пантеры', 'Полорогие', 'Собачие', 'Сурок']

# InTrain(df, a)
#
# for i in range(28015):
#     shutil.move(f'train/{df["image_name"].iloc[i]}',
#                 f'train2/{a[df["class_id"].iloc[i]]}')
#
# for i in os.listdir('train2'):
#     x = random.sample(os.listdir(f'train2/{i}'), len(os.listdir(f'train2/{i}')))
#     for j in range(len(x) // 10):
#         shutil.move(f'train2/{i}/{x[j]}', f'test/{i}/{x[j]}')