import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os
import keras
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
	#val_split_num = int(round(0.2*len(y)))
	#x_train = x[val_split_num:]
	#y_train = y[val_split_num:]
	#x_test = x[:val_split_num]
	#y_test = y[:val_split_num]

	image_size = 244

	# Generate dummy data
	x_train = np.random.random((100, image_size, image_size, 3))
	y_train = keras.utils.to_categorical(np.random.randint(2, size=(100, 1)), num_classes=2)
	x_test = np.random.random((20, image_size, image_size, 3))
	y_test = keras.utils.to_categorical(np.random.randint(2, size=(20, 1)), num_classes=2)

	print('x_train', x_train.shape)
	print('y_train', y_train.shape)
	print('x_test', x_test.shape)
	print('y_test', y_test.shape)

	img_rows, img_cols, img_channel = 224, 224, 3

	base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=x_train.shape[1:])

	add_model = Sequential()
	add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
	add_model.add(Dense(256, activation='relu'))
	add_model.add(Dense(1, activation='sigmoid'))

	model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
	model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
	              metrics=['accuracy'])

	model.summary()

	batch_size = 32
	epochs = 50

	train_datagen = ImageDataGenerator(
	        rotation_range=30, 
	        width_shift_range=0.1,
	        height_shift_range=0.1, 
	        horizontal_flip=True)

	train_datagen.fit(x_train)

	history = model.fit_generator(
	    train_datagen.flow(x_train, y_train, batch_size=batch_size),
	    steps_per_epoch=x_train.shape[0] // batch_size,
	    epochs=epochs,
	    validation_data=(x_test, y_test),
	    callbacks=[ModelCheckpoint('VGG16-transferlearning.model', monitor='val_acc', save_best_only=True)]
	   )