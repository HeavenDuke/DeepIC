import cv2
import numpy as np
from sklearn.cluster import k_means
from sklearn.preprocessing import normalize

from utils.loader import construct_input_data
from utils.preprocessor import shuffle


def imageSIFT(img, n_clusters = 100):
    s = cv2.SURF()
    print img
    pic = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = s.detectAndCompute(pic, None)
    descriptors = normalize(descriptors, norm = 'l2', axis = 1)
    return np.reshape(k_means(descriptors, n_clusters = n_clusters), newshape = (1, n_clusters * 128))


def extractSIFT(images):
    return np.asarray([imageSIFT(np.reshape(images[index], newshape = (64, 64, 3))) for index in range(images.shape[0])])

validation_split = 0.9

class_num = 12
enhanced_class_num = 10

x, y = construct_input_data('./data/train')
x_extra, y_extra = construct_input_data('./data/extra')

x, y = np.concatenate((x, x_extra)), np.concatenate((y, y_extra))

x, y = shuffle(x, y)

x_sift = extractSIFT(x)

print x_sift
