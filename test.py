import cv2
import numpy as np
from sklearn.cluster import k_means
from sklearn.preprocessing import normalize

from utils.loader import construct_input_data
from utils.preprocessor import shuffle


def imageSIFT(img, n_clusters = 50):
    s = cv2.SURF()
    pic = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = s.detectAndCompute(pic, None)
    descriptors = normalize(descriptors, norm = 'l2', axis = 1)
    centroid, l, i = k_means(descriptors, n_clusters = n_clusters)
    print centroid.shape
    return np.reshape(centroid, newshape = (1, n_clusters * 64))


def extractSIFT(images):
    return np.asarray([imageSIFT(images[index]) for index in range(len(images))])

validation_split = 0.9

class_num = 12
enhanced_class_num = 10

x, y = construct_input_data('./data/train')
x_extra, y_extra = construct_input_data('./data/extra')

x, y = x + x_extra, np.concatenate((y, y_extra))

x_sift = extractSIFT(x)

# x = np.asarray(x)
#
# x, y = shuffle(x, y)

print x_sift
