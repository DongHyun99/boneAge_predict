import cv2 as cv
import numpy as np

black = [1380, 1452, 1463, 1587, 1589, 1747, 1838, 1893, 1979, 2368]
avg = []
for b in black:
    img = cv.imread('./bone_data/train/'+str(b)+'.png', cv.IMREAD_GRAYSCALE)
    print(np.average(img.flatten()))
    avg.append(np.average(img.flatten()))

print('black avg: ', sum(avg)/10)