import pandas as pd
import matplotlib.pyplot as plt
from bone_data.DataLoad import train_data

#train_loss = pd.read_csv('runs/runs_Sep10_23-49-21_DESKTOP-8B4K2HF.csv')
#val_loss = pd.read_csv('runs/runs_Sep10_23-49-21_DESKTOP-8B4K2HF (1).csv')
#plt.plot(train_loss.Step,train_loss.Value, label='train_loss')
#plt.plot(val_loss.Step,val_loss.Value, label='validation_loss')
#plt.xlabel('epoch')
#plt.ylabel('mae loss')
#plt.legend()

plt.hist(train_data.male)
plt.xlabel('gender')
plt.ylabel('data')
plt.show()