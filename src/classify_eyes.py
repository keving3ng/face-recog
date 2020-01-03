import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PATH = "./data/glasses"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

train_glasses_dir = os.path.join(train_dir, "glasses")
train_regular_dir = os.path.join(train_dir, "regular")
validation_glasses_dir = os.path.join(validation_dir, "glasses")
validation_regular_dir = os.path.join(validation_dir, "regular")

def trainModel():

	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)),
		tf.keras.layers.MaxPooling2D(2, 2),
		tf.keras.layers.Flatten(), 
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')
	])

	model.summary()

	model.compile(loss='binary_crossentropy',
				  optimizer=RMSprop(lr=0.001),
				  metrics=['acc'])

	

if __name__ == "__main__":
	print("Tensorflow version= " + tf.__version__)
	trainModel()