from keras.layers import Conv2D, Activation
from keras.layers.merge import add


def InceptionBlockLayer(input, filter_sizes = (32, 32)):
    shortcut, residual = Conv2D()(input), None
    layer = shortcut
    for index in range(len(filter_sizes[1:])):
        layer = Activation(activation = "relu")(layer)
        if index == len(filter_sizes) - 2:
            residual = Conv2D()(layer)
            layer = residual
        else:
            layer = Conv2D()(layer)
    layer = add([shortcut, residual])
    return layer
