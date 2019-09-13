import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pickle

xpath = pickle.load(open('xset.pkl', 'rb'))
y = pickle.load(open('yset.pkl', 'rb'))

xset = []
for path in xpath:
    im = Image.open(path)
    xset.append(im)

xset = xset.astype(np.float32).reshape(-1, 128*128)/255.0
