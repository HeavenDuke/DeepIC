from utils.loader import construct_input_data
from keras.datasets import cifar10
from models.NaiveSPPNet import NaiveSPPNet
from utils.preprocessor import shuffle
from keras.preprocessing.image import ImageDataGenerator
from models.NaiveLeNet import NaiveLeNet

validation_split = 0.9

x, y = construct_input_data('./data')
x, y = shuffle(x, y)
x_train, y_train = x[:int(x.shape[0] * validation_split)], y[:int(x.shape[0] * validation_split)]
x_test, y_test = x[int(x.shape[0] * validation_split):], y[int(x.shape[0] * validation_split):]

generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False  # randomly flip images
)

generator.fit(x_train)

classifier = NaiveLeNet(class_num = 12)

classifier.fit_generator(generator.flow(x_train, y_train, batch_size = 32),
                         epochs = 500,
                         steps_per_epoch = x.shape[0] // 32,
                         validation_data = (x_test, y_test))
