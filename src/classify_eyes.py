import os

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PATH = "./data/glasses"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

train_glasses_dir = os.path.join(train_dir, "glasses")
train_regular_dir = os.path.join(train_dir, "reg")
validation_glasses_dir = os.path.join(validation_dir, "glasses")
validation_regular_dir = os.path.join(validation_dir, "reg")


print(len(os.listdir(train_glasses_dir)))
print(len(os.listdir(train_regular_dir)))
print(len(os.listdir(validation_glasses_dir)))
print(len(os.listdir(validation_regular_dir)))