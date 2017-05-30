import cv2
import os
import numpy as np


def construct_input_data(path, with_label = True):

    def vectorize(index):
        result = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        result[index - 1] = 1.
        return result

    image_ids = [int(f.replace('.jpg', '')) for f in os.listdir(path + '/image')]
    image_files = [cv2.resize(cv2.imread(path + "/image/" + str(f) + ".jpg", cv2.IMREAD_COLOR), (64, 64)) for f in
                   image_ids]

    if with_label:
        label_file = open(path + '/label.label', 'r')
        lines = label_file.readlines()
        labels_map = {}
        for line in lines:
            labels_map[int(line.split(" ")[0])] = vectorize(int(line.split(" ")[1].replace("\r", "").replace("\n", "")))
        image_labels = [labels_map[image_id] for image_id in image_ids]

        return np.asarray(image_files), np.asarray(image_labels)

    else:
        return image_ids, np.asarray(image_files), None
