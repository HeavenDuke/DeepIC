from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from layers.SpatialPyramidPooling import SpatialPyramidPooling


def ResSppNet(class_number):
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), strides = 3, padding = "same", input_shape = (3, None, None)))
    model.add(Activation(activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), strides = 3, padding = "same"))
    model.add(Activation(activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), strides = 3, padding = "same"))
    model.add(Activation(activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2), padding = "same"))
    model.add(SpatialPyramidPooling([1, 2, 4]))
    model.add(Flatten())
    model.add(Dense(units = class_number, kernel_regularizer = l2(l = 0.01)))
    model.add(Activation(activation = "sigmoid"))
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = 0.0001), metrics = ['accuracy'])
    return model
