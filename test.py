import numpy as np

from utils.loader import construct_input_data
from utils.preprocessor import shuffle, extractSIFT

validation_split = 0.9

class_num = 12
enhanced_class_num = 10

x, y = construct_input_data('./data/train')
x_extra, y_extra = construct_input_data('./data/extra')

x, y = np.concatenate((x, x_extra)), np.concatenate((y, y_extra))

x, y = shuffle(x, y)

x_sift = extractSIFT(x)
