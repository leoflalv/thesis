import os
import numpy as np
from keras import backend as keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D


def alexnet(pretrained_weights=None, input_size=(17, 17, 3), padding='same'):
    model = Sequential()

    print('first')
    # 1st Convolutional Layer
    model.add(Conv2D(filters=8, input_shape=input_size,
                     kernel_size=(3, 3), strides=(1, 1), padding=padding))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           padding=padding))

    print('second')
    # 2nd Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(
        3, 3), strides=(1, 1), padding=padding))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           padding=padding))

    print('third')
    # 3rd Convolutional Layer
    model.add(Conv2D(filters=32, kernel_size=(
        3, 3), strides=(1, 1), padding=padding))
    model.add(Activation('relu'))

    print('fourth')
    # 4th Convolutional Layer
    model.add(Conv2D(filters=32, kernel_size=(
        3, 3), strides=(1, 1), padding=padding))
    model.add(Activation('relu'))

    print('fifth')
    # 5th Convolutional Layer
    model.add(Conv2D(filters=16, kernel_size=(
        3, 3), strides=(1, 1), padding=padding))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),
                           padding=padding))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(1024, input_shape=(
        input_size[0]*input_size[1]*input_size[2],)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # # 2nd Fully Connected Layer
    # model.add(Dense(4096))
    # model.add(Activation('relu'))
    # # Add Dropout
    # model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(512))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def lenet(pretrained_weights=None, input_size=(13, 13, 3)):
    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(3, 3),
                     activation='relu', input_shape=input_size))
    model.add(MaxPooling2D())

    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(units=200, activation='relu'))
    model.add(Dense(units=84, activation='relu'))

    model.add(Dense(units=1, activation='sigmoid'))

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
