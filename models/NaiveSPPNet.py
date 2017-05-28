from keras.layers import Dense, Flatten, Conv2D, Convolution2D, MaxPooling2D, Activation, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from layers.SpatialPyramidPooling import SpatialPyramidPooling
from keras.models import Sequential


def NaiveSPPNet(class_num):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode = "same", input_shape = (3, None, None)))
    model.add(Activation(activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Convolution2D(32, 3, 3, border_mode = "same"))
    model.add(Activation(activation = "relu"))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Convolution2D(32, 3, 3, border_mode = "same"))
    model.add(Activation(activation = "relu"))
    model.add(SpatialPyramidPooling([1, 2, 4]))
    model.add(Dense(units = class_num, kernel_regularizer = l2(l = 0.01)))
    model.add(Activation(activation = "sigmoid"))
    model.compile(optimizer = Adam(lr = 1e-3), loss = "binary_crossentropy", metrics = ['accuracy'])
    return model
