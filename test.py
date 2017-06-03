import numpy as np

from utils.loader import construct_input_data
from utils.preprocessor import group_data_by_label

validation_split = 0.8

class_num = 12
enhanced_class_num = 10

x, y = construct_input_data('./data/train', with_masks = False)
x_extra, y_extra = construct_input_data('./data/extra', with_masks = False)

x, y = x + x_extra, np.concatenate((y, y_extra))

x, y = x.astype(np.float32), y.astype(np.float32)

x /= 255.

table = group_data_by_label(x, y)

for key in table:
    print table[key]["labels"].shape
    print table[key]["images"].shape