# CAPTCHA RECOGNITION WITH ATTACHED BINARY IMAGES (CRABIs)

- Author: HOANG NGOC LINH
- Email: linhhn.bkdn@gmail.com
- Phone: +84 344 965 661

# Model: "sequential"
Layer (type)                 Output Shape              Param #
conv2d (Conv2D)              (None, 320, 96, 32)       320
max_pooling2d (MaxPooling2D) (None, 160, 48, 32)       0
conv2d_1 (Conv2D)            (None, 160, 48, 64)       18496
max_pooling2d_1 (MaxPooling2 (None, 80, 24, 64)        0
conv2d_2 (Conv2D)            (None, 80, 24, 128)       73856
conv2d_3 (Conv2D)            (None, 80, 24, 64)        8256
conv2d_4 (Conv2D)            (None, 80, 24, 128)       73856
max_pooling2d_2 (MaxPooling2 (None, 40, 12, 128)       0
conv2d_5 (Conv2D)            (None, 40, 12, 256)       295168
conv2d_6 (Conv2D)            (None, 40, 12, 128)       32896
conv2d_7 (Conv2D)            (None, 40, 12, 256)       295168
conv2d_8 (Conv2D)            (None, 40, 12, 128)       32896
conv2d_9 (Conv2D)            (None, 40, 12, 256)       295168
max_pooling2d_3 (MaxPooling2 (None, 20, 6, 256)        0
conv2d_10 (Conv2D)           (None, 20, 6, 512)        1180160
conv2d_11 (Conv2D)           (None, 20, 6, 256)        131328
conv2d_12 (Conv2D)           (None, 20, 6, 512)        1180160
conv2d_13 (Conv2D)           (None, 20, 6, 256)        131328
conv2d_14 (Conv2D)           (None, 20, 6, 512)        1180160
conv2d_15 (Conv2D)           (None, 20, 6, 256)        131328
conv2d_16 (Conv2D)           (None, 20, 6, 512)        1180160
max_pooling2d_4 (MaxPooling2 (None, 10, 3, 512)        0
flatten (Flatten)            (None, 15360)             0
dropout (Dropout)            (None, 15360)             0
dense (Dense)                (None, 5)                 76805
activation (Activation)      (None, 5)                 0

Total params: 6,317,509
Trainable params: 6,317,509
Non-trainable params: 0

# Validation dataset result:
Epoch 88/200
419/419 [==============================] - 51s 120ms/step - loss: 0.0157 - accuracy: 0.9948 - val_loss: 1.8523 - val_accuracy: 0.7045

Epoch 89/200
419/419 [==============================] - 50s 120ms/step - loss: 0.0176 - accuracy: 0.9938 - val_loss: 1.8227 - val_accuracy: 0.7098
