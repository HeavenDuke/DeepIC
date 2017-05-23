import cv2
import os


def construct_input_data(path):

    def vectorize(index):
        result = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        result[index - 1] = 1.
        return result

    label_file = open(path + 'data/train.label', 'r')
    image_ids = [int(f.replace('.jpg', '')) for f in os.listdir(path + 'data/image')]
    image_files = [cv2.imread(path + "data/image/" + str(f) + ".jpg", cv2.IMREAD_COLOR) / 255. for f in image_ids]
    lines = label_file.readlines()
    labels_map = {}
    for line in lines:
        labels_map[int(line.split(" ")[0])] = vectorize(int(line.split(" ")[1].replace("\r", "").replace("\n", "")))
    image_labels = [labels_map[image_id] for image_id in image_ids]
    return image_files, image_labels
