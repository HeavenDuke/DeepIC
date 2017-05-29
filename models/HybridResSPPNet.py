from keras.models import Model
from keras.layers import Dense, ZeroPadding2D, AveragePooling2D, BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Input, Concatenate
from keras.optimizers import RMSprop
from layers.SpatialPyramidPooling import SpatialPyramidPooling
from keras.applications.resnet50 import identity_block, conv_block


def EnhancedHybridResSppNet(class_num, enhanced_class_num):
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

    model = MaxPooling2D((2, 2))(model)

    model = SpatialPyramidPooling([1, 2, 4])(model)

    model1 = Dense(units = class_num)(model)
    model1 = Activation(activation = "softmax")(model1)
    model1 = Model(_input, model1)
    model1.compile(loss = "categorical_crossentropy", optimizer = RMSprop(lr = 1e-4, decay = 1e-6), metrics = ['accuracy'])

    model2 = Dense(units = enhanced_class_num)(model)
    model2 = Activation(activation = "softmax")(model2)
    model2 = Model(_input, model2)
    model2.compile(loss = "categorical_crossentropy", optimizer = RMSprop(lr = 1e-4, decay = 1e-6), metrics = ['accuracy'])

    input2 = Input(shape = (128 * 150))

    model3 = Concatenate((input2, model))
    model3 = Dense(units = class_num)(model3)
    model3 = Activation(activation = "softmax")(model3)
    model3 = Model(inputs = [_input, input2], outputs = model3)
    model3.compile(loss = "categorical_crossentropy", optimizer = RMSprop(lr = 1e-4, decay = 1e-6), metrics = ['accuracy'])

    return model1, model2, model3
