from models.ResSppNet import EnhancedResSppNet
from utils.loader import construct_input_data
from utils.writer import save_prediction
import numpy as np

ids, x_test = construct_input_data('./data/test', with_label = False)

classifier, classifier_p, classifier_e = EnhancedResSppNet(class_num = 12, enhanced_class_num = 10)

classifier_e.load_weights('./weights/ResSppNet.h5')

y_predict = np.argmax(classifier_e.predict(x_test), axis = 1)

save_prediction(ids, y_predict, './data/test/prediction.label')
