import cv2
import os
import numpy as np
from preprocessor import removeBackground


def construct_input_data(path, with_masks = True, with_label = True):

    def vectorize(index):
        result = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        result[index - 1] = 1.
        return result

    image_ids = [int(f.replace('.jpg', '')) for f in os.listdir(path + '/image')]
    image_files = [cv2.imread(path + "/image/" + str(f) + ".jpg", cv2.IMREAD_COLOR) for f in image_ids]

    if with_masks:
        mask_files = [cv2.imread(path + "/mask/" + str(f) + ".png", cv2.IMREAD_COLOR) for f in image_ids]
        image_files = [removeBackground(image_files[index], mask_files[index]) for index in range(len(mask_files))]

    if with_label:
        label_file = open(path + '/label.label', 'r')
        lines = label_file.readlines()
        labels_map = {}
        for line in lines:
            labels_map[int(line.split(" ")[0])] = vectorize(int(line.split(" ")[1].replace("\r", "").replace("\n", "")))
        image_labels = [labels_map[image_id] for image_id in image_ids]

        return image_files, np.asarray(image_labels)

    else:
        return image_ids, image_files, None
