import matplotlib.pyplot as plt
import pandas as pd

loss = pd.read_csv('result/loss그래프.csv')
loss_list=loss.loss
val_loss_list = loss.val_loss

plt.figure()
plt.plot([x for x in range(1,21,1)], loss_list, label='loss')
    
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()


plt.plot([x for x in range(1,21,1)], val_loss_list, label='validation_loss')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.show()