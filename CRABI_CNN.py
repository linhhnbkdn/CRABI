#usr/bin python3
import os
import logging

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential


class CRABI_CNN:
    logger = logging.getLogger(name='CRABI_CNN')
    @staticmethod
    def build(rows, cols, channels, classes, **kwargs):
        inputShape = (rows, cols, channels)
        # if we are using "channels first",
        # update the input
        if K.image_data_format() == "channels_first":
            inputShape = (channels, rows, cols)
        CRABI_CNN.logger.info("Input Shape: {}".format(inputShape))
        # Build model
        model = Sequential()
        # CON2D --> MaxPool2D
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                            input_shape=inputShape))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        # Conv2D ---> MaxPool2D
        model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        # Conv2D --> Conv2D --> Conv2D --> MaxPool2D
        model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
        model.add(Conv2D(64, kernel_size=(1, 1), padding='same'))
        model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        # Conv2D --> Conv2D --> Conv2D --> Conv2D --> Conv2D --> MaxPool2D
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
        model.add(Conv2D(128, kernel_size=(1, 1), padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
        model.add(Conv2D(128, kernel_size=(1, 1), padding='same'))
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))
        # Conv2D --> Conv2D --> Conv2D --> Conv2D --> Conv2D --> Conv2D
        # --> Conv2D --> MaxPool2D
        model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
        model.add(Conv2D(256, kernel_size=(1, 1), padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
        model.add(Conv2D(256, kernel_size=(1, 1), padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
        model.add(Conv2D(256, kernel_size=(1, 1), padding='same'))
        model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=2))

        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        model.summary()
        return model



model = CRABI_CNN.build(96, 320, 3, 62)