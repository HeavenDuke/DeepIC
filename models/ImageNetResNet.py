from keras.applications import resnet50
from keras.layers import Flatten, Dense
from keras.regularizers import l2


def EnhancedImageNetResNset(class_num):
    pretrained_model = resnet50.ResNet50(include_top = True, weights = 'imagenet', classes = 1000, input_shape = (128, 128, 32))

    layer = pretrained_model.get_layer(name = "avg_pool")

    model = Flatten()(layer)

    model = Dense(units = class_num, kernel_regularizer = l2(l = 0.01))(model)

    return model
