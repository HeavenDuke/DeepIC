# import numpy as np
#
# from utils.loader import construct_input_data
# from utils.preprocessor import extractSIFT
#
# validation_split = 0.8
#
# class_num = 12
# enhanced_class_num = 10
#
# x, y = construct_input_data('./data/train', with_masks = False)
# x_extra, y_extra = construct_input_data('./data/extra', with_masks = False)
#
# x, y = x + x_extra, np.concatenate((y, y_extra))
#
# x_sift = extractSIFT(x, n_clusters = 100)
#
# print x_sift

# x = np.asarray([np.reshape(cv2.resize(item, (128, 128)), newshape = (128, 128, 3)) for item in x])
#
# x, y = x.astype(np.float32), y.astype(np.float32)
#
# x /= 255.
#
#
from sklearn.cluster import k_means
import numpy as np
n_clusters = 100
print k_means(np.random.randint(0, 100, size = (10000, 25)), n_clusters = n_clusters, verbose = True, max_iter = 1, n_init = 1)
