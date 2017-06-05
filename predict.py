from keras.optimizers import RMSprop

from models.ResSppNet import ResnetBuilder
from utils.loader import construct_input_data
from utils.preprocessor import extractSIFT
from utils.writer import save_prediction
import numpy as np
import cv2

ids, x_test = construct_input_data('./data/extra', with_label = False)

# x_sift = extractSIFT(x_test)

x_test = np.asarray([cv2.resize(item, (128, 128)) for item in x_test])

x_test = x_test.astype(np.float32)

x_test /= 255.

classifier = ResnetBuilder.build_resnet_34(input_shape = (3, 128, 128), num_outputs = 12, enhanced = False)

classifier.compile(loss = "categorical_crossentropy", optimizer = RMSprop(lr = 5e-4, decay = 1e-3),
                   metrics = ['accuracy'])

classifier.load_weights('./weights/ResSppNet.h5')

y_predict = np.argmax(classifier.predict(x_test), axis = 1)

save_prediction(ids, y_predict, './data/test/prediction.label')
