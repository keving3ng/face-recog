#!env/bin/python
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from crop_eye import find_eye

DATA_PATH = "./data/glasses-detection"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def trainModel():
	IMG_SIZE = (4, 4)
	class trainingCallback(tf.keras.callbacks.Callback):
		def on_epoch_end(self, epoch, logs={}, threshold=0.99):
			if(logs.get('acc') >= threshold):
				self.model.stop_training = True

	train_glasses_dir = os.path.join(DATA_PATH, "train/glasses")
	train_regular_dir = os.path.join(DATA_PATH, "train/regular")
	validation_glasses_dir = os.path.join(DATA_PATH, "validation/glasses")
	validation_regular_dir = os.path.join(DATA_PATH, "validation/regular")

	callbacks = trainingCallback()

	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(4, 4, 3)),
		tf.keras.layers.MaxPooling2D(2, 2),
		tf.keras.layers.Flatten(), 
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(1, activation='sigmoid')
	])

	model.summary()

	model.compile(loss='binary_crossentropy',
				  optimizer=RMSprop(lr=0.001),
				  metrics=['acc'])

	# preprocessing_function=find_eye,
	train_datagen = ImageDataGenerator(rescale=1/255)
	validation_datagen = ImageDataGenerator(rescale=1/255)

	train_generator = train_datagen.flow_from_directory(
		os.path.join(DATA_PATH, "train"),
		target_size=IMG_SIZE,
		batch_size = 4, 
		class_mode='binary'
	)

	validation_generator = validation_datagen.flow_from_directory(
		os.path.join(DATA_PATH, "validation"),
		target_size=IMG_SIZE,
		batch_size = 4, 
		class_mode='binary'
	)

	history = model.fit(
		train_generator,
		epochs=10,
		callbacks=[callbacks],
		verbose=1,
		validation_data = validation_generator,
		validation_steps = 2
	)

if __name__ == "__main__":
	print("Tensorflow version= " + tf.__version__)
	trainModel()
