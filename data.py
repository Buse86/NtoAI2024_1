import pandas as pd
import shutil
df = pd.read_csv('train.csv')
df = df.drop(columns=['unified_class'])

a = ['Заяц', 'Кабан', 'Кошки',  'Куньи', 'Медведь', 'Оленевые', 'Пантеры', 'Полорогие', 'Собачие', 'Сурок']

for i in range(28015):
    shutil.move(f'C:/Users/Geo/PycharmProjects/nto2024AI_1/train/{df["image_name"].iloc[i]}', f'C:/Users/Geo/PycharmProjects/nto2024AI_1/train2/{a[df["class_id"].iloc[i]]}')