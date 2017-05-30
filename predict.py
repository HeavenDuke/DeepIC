from models.ResSppNet import EnhancedResSppNet
from utils.loader import construct_input_data
from utils.preprocessor import extractSIFT
from utils.writer import save_prediction
import numpy as np
import cv2

ids, x_test = construct_input_data('./data/test', with_label = False)

x_sift = extractSIFT(x_test)

x_test = np.asarray([cv2.resize(item, (128, 128)) for item in x_test])

x_test = x_test.astype(np.float32)

x_test /= 255.

classifier, classifier_p, classifier_e = EnhancedResSppNet(class_num = 12, enhanced_class_num = 10)

classifier_e.load_weights('./weights/ResSppNet.h5')

y_predict = np.argmax(classifier_e.predict(x_test), axis = 1)

save_prediction(ids, y_predict, './data/test/prediction.label')
