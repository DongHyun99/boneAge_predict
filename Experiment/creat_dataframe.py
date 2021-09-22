import numpy as np
import pandas as pd

#data load
train_img_path = 'bone_data/train/'
train_csv_path = 'bone_data/training_dataset.csv'

# dataset setting
train_data = pd.read_csv(train_csv_path)
train_data.iloc[:, 1:3] = train_data.iloc[:, 1:3].astype(np.float)
df = []

t1 = train_data[train_data['boneage']<57]
t2 = train_data[(train_data['boneage']>=57) & (train_data['boneage']<=114)]
t3 = train_data[(train_data['boneage']>114) & (train_data['boneage']<=171)]
t4 = train_data[(train_data['boneage']>171) & (train_data['boneage']<=228)]


angles = [-15, -10, -5, 0, 5, 10, 15]

for angle in angles:
    for tlanslate in ['original', 'right50', 'left50']:
        data = train_data.copy()
        data['id']=data['id'].apply(lambda x: str(x)+'-{}-rotate{}'.format(tlanslate, angle))
        df.append(data)
new_train = df.pop(0)
for data in df:
    new_train = pd.concat([new_train, data])

print(new_train)

new_train.to_csv('D:/new_train.csv',sep=',', index = None, na_rep='NaN')