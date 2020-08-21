import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow.keras.utils import to_categorical
import random,shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from google.colab import drive
drive.mount('/content/drive')

train_image_gen = ImageDataGenerator(
    rescale = 1/255.0,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.3,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_gen = train_image_gen.flow_from_directory(
    "/content/drive/My Drive/data/train",
    target_size = ((24,24)),
    batch_size = 64,
    class_mode = 'categorical',
    color_mode='grayscale'
)

test_image_gen = ImageDataGenerator(
    rescale = 1/255.0,
)

test_gen = test_image_gen.flow_from_directory(
    "/content/drive/My Drive/data/test",
    target_size = ((24,24)),
    batch_size = 16,
    class_mode = 'categorical',
    color_mode='grayscale'
)

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(24,24,1)))  #no. of filters, kernel size
model.add(BatchNormalization()) #mean 0 - standard deviation 1
model.add(MaxPooling2D(2,2))  #Downsamples by taking the maximum value over the window defined by pool_size, reduces the size, increases speed
model.add(Dropout(0.4)) #reduces overfitting and increases generalization
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))  #Fraction of the input units to drop
model.add(Flatten())
model.add(Dense(32,activation='relu'))  #no. of units
model.add(Dropout(0.4))
model.add(Dense(2,activation='softmax'))
model.summary()
# 52,562 parameters out of which 192 non trainable parameters
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

#Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data.

model.fit_generator(train_gen, validation_data=test_gen,epochs=5,steps_per_epoch=4321//64,validation_steps=525//16)

model.save('../content/models/cnnCat2.h5', overwrite=True)

