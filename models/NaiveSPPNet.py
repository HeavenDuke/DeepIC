from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
from keras.regularizers import l2
from layers.SpatialPyramidPooling import SpatialPyramidPooling


def NaiveSPPNet(x, class_num):
    model = x
    model = Conv2D(filters = 32, kernel_size = (3, 3), strides = 3, padding = "same", input_shape = (3, None, None))(model)
    model = Activation(activation = "relu")(model)
    model = MaxPooling2D(pool_size = (2, 2), padding = "same")(model)
    model = Conv2D(filters = 32, kernel_size = (3, 3), strides = 3, padding = "same")(model)
    model = Activation(activation = "relu")(model)
    model = MaxPooling2D(pool_size = (2, 2), padding = "same")(model)
    model = Conv2D(filters = 32, kernel_size = (3, 3), strides = 3, padding = "same")(model)
    model = Activation(activation = "relu")(model)
    model = MaxPooling2D(pool_size = (2, 2), padding = "same")(model)
    model = SpatialPyramidPooling([1, 2, 4])(model)
    model = Flatten()(model)
    model = Dense(units = class_num, kernel_regularizer = l2(l = 0.01))(model)
    model = Activation(activation = "sigmoid")(model)
    return model
