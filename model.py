from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential


def CNNmodel(model_output_size,image_height,image_width):
    model = Sequential()
    model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu',input_shape = (image_height,image_width,3)))
    model.add((Conv2D(filters = 64, kernel_size=(3,3),activation='relu')))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256,activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(model_output_size,activation='softmax'))
    model.summary()
    return model

