import cv2 
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense



batch_size=15

# train_datagen is used to preprocess the image. 
train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagen = ImageDataGenerator(
	rescale=1./255
	)

train_generator = train_datagen.flow_from_directory(
	'data/train',
	target_size=(150,150),
	batch_size=batch_size,
    class_mode = 'categorical'
	)


validation_generator = test_datagen.flow_from_directory(
	'data/validation',
	target_size=(150,150),
	batch_size=batch_size,
    class_mode = 'categorical'
	)


cnn_model = keras.Sequential([

	keras.layers.Conv2D(32, (3,3), strides=1, padding='same', input_shape=(150,150,3)),
	keras.layers.Activation('relu'),
	keras.layers.MaxPooling2D(pool_size=(2,2)),


	keras.layers.Conv2D(32, (3,3), strides=1, padding='same', input_shape=(150,150,3)),
	keras.layers.Activation('relu'),
	keras.layers.MaxPooling2D(pool_size=(2,2)),

	keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
	keras.layers.Dense(58,activation = 'softmax')
	])

cnn_model.summary()



cnn_model.compile(optimizer=tf.train.AdamOptimizer(),
	loss='categorical_crossentropy',
	metrics=['accuracy'])


cnn_model.fit_generator(
	train_generator,
	steps_per_epoch=290,#	number of training example photos
	epochs=3,
	validation_data=validation_generator,
	validation_steps=14 #number of testing photos
	)

from keras.preprocessing import image
test_image = image.load_img('/home/icts/practice-datasets/OCR/data/validation/img062-00004.png', target_size = (150,150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn_model.predict(test_image)
print np.argmax(result)
