#!env/bin/python
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from crop_eye import find_eye

DATA_PATH = "./data/glasses"
train_dir = os.path.join(DATA_PATH, "train")
validation_dir = os.path.join(DATA_PATH, "validation")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class trainCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		if(logs.get['acc'] >= 0.95):
			self.model.stop_training = True

def trainModel():
	

	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
		tf.keras.layers.MaxPooling2D(2, 2),
		tf.keras.layers.Flatten(), 
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')
	])

	model.summary()

	model.compile(loss='binary_crossentropy',
				  optimizer=RMSprop(lr=0.001),
				  metrics=['acc'])

	history = model.fit_generator(
		train_generator,
		steps_per_epoch=8, 
		epochs=10,
		verbose=1
	)

if __name__ == "__main__":
	print("Tensorflow version= " + tf.__version__)
	trainModel()
