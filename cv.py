import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

black = [1380, 1452, 1463, 1587, 1589, 1747, 1838, 1893, 1979, 2368]
avg = []
for b in black:
    img = cv.imread('./bone_data/train/'+str(b)+'.png', cv.IMREAD_GRAYSCALE)
    print(np.average(img.flatten()))
    avg.append(np.average(img.flatten()))

print('black avg: ', sum(avg)/10)


TRAIN_PATH = glob.glob('bone_data/train/'+'*.png')
c = []

for i in TRAIN_PATH:
    img = cv.imread(i, cv.IMREAD_GRAYSCALE)
    c.append(np.average(img.flatten()))

plt.hist(c)
plt.show()

'''
~24: 750장
25~41: 7160장
43~61:2470장
62~80: 815장
80~99: 1030장
99~: 375장
'''