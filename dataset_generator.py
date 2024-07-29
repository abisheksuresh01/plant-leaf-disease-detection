import os
import cv2
import random


dir = "../plant-disease-detection/PlantVillage-Dataset/raw/segmented"
arr = os.listdir(dir)
healthy = []
unhealthy = []

for a in arr:
    if a == 'hist.py':
        continue
    a1, a2 = a.split('___')
    disease_type = a2.replace(a1 + '_', '')
    is_h = 1 if 'healthy' in a else 0
    if is_h:
        healthy = healthy + list(map(lambda x: a + '/' + x, os.listdir(dir + '/' + a)))
    else:
        unhealthy = unhealthy + list(map(lambda x: a + '/' + x, os.listdir(dir + '/' + a)))

random.seed(94827263283)
random.shuffle(unhealthy)
unhealthy = unhealthy[:len(healthy)]

open('healthy.txt', 'w').writelines(map(lambda x: dir + '/' + x + '\n', healthy))
open('unhealthy.txt', 'w').writelines(map(lambda x: dir + '/' + x + '\n', unhealthy))
