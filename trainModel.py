import os
import logging
import random

import captcha
import numpy as np
import cv2
from captcha.image import ImageCaptcha

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import  ModelCheckpoint
from tensorflow.keras.preprocessing import image_dataset_from_directory

from CRABI_CNN import CRABI_CNN


os.environ['DISPLAY'] = ":0"
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s %(name)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


logger = logging.getLogger("I'm {}".format(str(__file__).split(os.sep)[-1]))
logger.setLevel(logging.DEBUG)

f_data = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
weights = os.path.join(f_data, 'weights.h5')
inputShape = (96, 320)


trainDataSet = image_dataset_from_directory(directory=f_data, color_mode='grayscale',
                                      label_mode='categorical', batch_size=32,
                                     image_size=inputShape, validation_split=0.33,
                                     subset="training", seed=123)
valDataSet = image_dataset_from_directory(directory=f_data, color_mode='grayscale',
                                      label_mode='categorical', batch_size=32,
                                     image_size=inputShape, validation_split=0.33,
                                     subset="validation", seed=123)


model = CRABI_CNN.build(inputShape[1], inputShape[0], 1, 5)
model.compile(optimizer=Adam(learning_rate=0.00001),
                loss=categorical_crossentropy,
                metrics=['accuracy'])


if os.path.exists(weights):
    model.load_weights(weights)

checkpoint = ModelCheckpoint(weights, monitor = 'val_accuracy',
                                verbose = 1, save_best_only = True,
                                mode = 'max')

history = model.fit(trainDataSet, validation_data=valDataSet,
          batch_size=64,
          epochs=200,  callbacks = [checkpoint])