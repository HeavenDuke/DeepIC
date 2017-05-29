from keras.models import Model
from keras.layers import Dense, ZeroPadding2D, AveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Input
from keras.optimizers import RMSprop
from layers.SpatialPyramidPooling import SpatialPyramidPooling
from keras.applications.resnet50 import identity_block, conv_block


def ResSppNet(class_number):
    _input = Input(shape = (None, None, 3))
    model = _input
    model = ZeroPadding2D((3, 3))(model)
    model = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(model)
    model = BatchNormalization(axis = 3, name = 'bn_conv1')(model)
    model = Activation('relu')(model)
    model = MaxPooling2D((3, 3), strides = (2, 2))(model)

    model = conv_block(model, 3, [64, 64, 256], stage = 2, block = 'a', strides = (1, 1))
    model = identity_block(model, 3, [64, 64, 256], stage = 2, block = 'b')
    model = identity_block(model, 3, [64, 64, 256], stage = 2, block = 'c')

    model = conv_block(model, 3, [128, 128, 512], stage = 3, block = 'a')
    model = identity_block(model, 3, [128, 128, 512], stage = 3, block = 'b')
    model = identity_block(model, 3, [128, 128, 512], stage = 3, block = 'c')
    model = identity_block(model, 3, [128, 128, 512], stage = 3, block = 'd')

    model = conv_block(model, 3, [256, 256, 1024], stage = 4, block = 'a')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'b')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'c')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'd')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'e')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'f')

    model = conv_block(model, 3, [512, 512, 2048], stage = 5, block = 'a')
    model = identity_block(model, 3, [512, 512, 2048], stage = 5, block = 'b')
    model = identity_block(model, 3, [512, 512, 2048], stage = 5, block = 'c')

    model = AveragePooling2D((7, 7))(model)

    model = SpatialPyramidPooling([1, 2, 4])(model)

    model = Dense(units = class_number)(model)
    model = Activation(activation = "softmax")(model)

    model = Model(_input, model)

    model.compile(loss = "categorical_crossentropy", optimizer = RMSprop(lr = 1e-4, decay = 1e-6), metrics = ['accuracy'])
    return model


def EnhancedResSppNet(class_num, enhanced_class_num):
    _input = Input(shape = (None, None, 3))
    model = _input
    model = ZeroPadding2D((3, 3))(model)
    model = Conv2D(64, (7, 7), strides = (2, 2))(model)
    model = BatchNormalization(axis = 3)(model)
    model = Activation('relu')(model)
    model = MaxPooling2D((3, 3), strides = (2, 2))(model)

    model = conv_block(model, 3, [64, 64, 256], stage = 2, block = 'a', strides = (1, 1))
    model = identity_block(model, 3, [64, 64, 256], stage = 2, block = 'b')
    model = identity_block(model, 3, [64, 64, 256], stage = 2, block = 'c')

    model = conv_block(model, 3, [128, 128, 512], stage = 3, block = 'a')
    model = identity_block(model, 3, [128, 128, 512], stage = 3, block = 'b')
    model = identity_block(model, 3, [128, 128, 512], stage = 3, block = 'c')
    model = identity_block(model, 3, [128, 128, 512], stage = 3, block = 'd')

    model = conv_block(model, 3, [256, 256, 1024], stage = 4, block = 'a')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'b')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'c')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'd')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'e')
    model = identity_block(model, 3, [256, 256, 1024], stage = 4, block = 'f')

    model = conv_block(model, 3, [512, 512, 2048], stage = 5, block = 'a')
    model = identity_block(model, 3, [512, 512, 2048], stage = 5, block = 'b')
    model = identity_block(model, 3, [512, 512, 2048], stage = 5, block = 'c')

    # model = AveragePooling2D((7, 7))(model)

    model = SpatialPyramidPooling([1, 2, 4])(model)

    # model = Flatten()(model)

    model1 = Dense(units = class_num)(model)
    model1 = Activation(activation = "softmax")(model1)
    model1 = Model(_input, model1)
    model1.compile(loss = "categorical_crossentropy", optimizer = RMSprop(lr = 1e-4, decay = 1e-6), metrics = ['accuracy'])

    model2 = Dense(units = enhanced_class_num)(model)
    model2 = Activation(activation = "softmax")(model2)
    model2 = Model(_input, model2)
    model2.compile(loss = "categorical_crossentropy", optimizer = RMSprop(lr = 1e-4, decay = 1e-6), metrics = ['accuracy'])

    return model1, model2
