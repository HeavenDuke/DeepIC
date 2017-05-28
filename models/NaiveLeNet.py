from keras.models import Sequential
from keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D, Activation, Dropout
from keras.optimizers import Adam


def NaiveLeNet(class_num):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = (64, 64, 3)))
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

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))
    model.compile(optimizer = Adam(lr = 1e-3), loss = "categorical_crossentropy", metrics = ['accuracy'])
    return model
