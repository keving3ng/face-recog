import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from crop_eye import DetectGlasses

PATH = "./data/glasses"
train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# train_glasses_dir = os.path.join(train_dir, "glasses")
# train_regular_dir = os.path.join(train_dir, "regular")
# validation_glasses_dir = os.path.join(validation_dir, "glasses")
# validation_regular_dir = os.path.join(validation_dir, "regular")

def trainModel():
	# Rescale all images by 1./255
	train_datagen = ImageDataGenerator(rescale=1/255)

	train_generator = train_datagen.flow_from_directory(
		train_dir,
		target_size=(32, 32),
		batch_size=1,
		class_mode='binary'
	)

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
