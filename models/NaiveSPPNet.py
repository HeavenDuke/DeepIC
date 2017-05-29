from keras.layers import Dense, Convolution2D, MaxPooling2D, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import RMSprop
from layers.SpatialPyramidPooling import SpatialPyramidPooling
from keras.models import Sequential


def NaiveSPPNet(class_num):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = (None, None, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(SpatialPyramidPooling([1, 2, 4]))
    model.add(Dense(units = class_num, kernel_regularizer = l2(l = 0.01)))
    model.add(Activation(activation = "sigmoid"))
    model.compile(optimizer = RMSprop(lr = 1e-3), loss = "binary_crossentropy", metrics = ['accuracy'])
    return model


def EnhancedNaiveSPPNet(class_num, enhanced_class_num):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = (None, None, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode = 'same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))

    model.add(SpatialPyramidPooling([1, 2, 4]))

    model1, model2 = Sequential(), Sequential()
    model1.add(model)
    model2.add(model)

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model1.add(Dense(class_num))
    model1.add(Activation('softmax'))
    model1.compile(optimizer = RMSprop(lr = 1e-4, decay = 1e-6), loss = "categorical_crossentropy", metrics = ['accuracy'])

    model2.add(Dense(enhanced_class_num))
    model2.add(Activation('softmax'))
    model2.compile(optimizer = RMSprop(lr = 1e-4, decay = 1e-6), loss = "categorical_crossentropy", metrics = ['accuracy'])

    return model1, model2
