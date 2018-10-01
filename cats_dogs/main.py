import cv2
from keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

image = cv2.imread("/home/icts/practice-datasets/cats_dogs/data/train/cat/cat.8.jpg")
#cv2.imshow('Loaded image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

x = img_to_array(image)
x = x.reshape((1,) + x.shape)


datagenerator = ImageDataGenerator(
	rotation_range=40,
	width_shift_range=0.2,
	height_shift_range=0.2,
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest')



i = 0
for a in datagenerator.flow(x, batch_size=1,
                          save_to_dir='/home/icts/practice-datasets/cats_dogs/data/preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))


model.add(Conv2D(32, (3,3), input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), dim_ordering="th"))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

batch_size = 16

train_datagenerator = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True)

test_datagenerator = ImageDataGenerator(rescale=1./255)

train_datagenerator = train_datagenerator.flow_from_directory(
	'data/train',
	target_size=(150, 150),
	batch_size=batch_size,
	class_mode='binary')

validation_generator = test_datagenerator.flow_from_directory(
	'data/validation',
	target_size=(150, 150),
	batch_size=batch_size,
	class_mode='binary')

model.fit_generator(
	train_datagenerator,
	steps_per_epoch=2000,
	epochs=5,
	validation_data=validation_generator,
	validation_steps=800
	)

model.save_weights('first.h5')