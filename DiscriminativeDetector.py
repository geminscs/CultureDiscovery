__author__ = 'admin'
import pickle
import numpy as np


class DiscriminativeDetector(object):
    classifiers = []
    params = dict()

    @classmethod
    def initialize(cls, patch_size=80, bins=10, feature_len=800):
        cls.params["patch_size"] = patch_size
        cls.params["bins"] = bins
        cls.params["feature_len"] = feature_len

    @classmethod
    def add_classifier(cls, path):
        f = open(path, "rb")
        classifier = pickle.load(f)
        f.close()
        for i in classifier:
            if i != 0:
                cls.classifiers.append(i)

    @classmethod
    def gen_dis_map(cls, image_feature_map, classifier):
        patch_size = cls.params["patch_size"] / cls.params["bins"]
        image_shape = image_feature_map.shape
        dis_mat = np.zeros(((image_shape[0] - patch_size + 1)*(image_shape[1] - patch_size + 1),
                            cls.params["feature_len"]))
        count = 0
        for i in range(0, image_shape[0] - patch_size + 1):
            for j in range(0, image_shape[0] - patch_size + 1):
                image_part = image_feature_map[i:i + patch_size, j:j + patch_size, :]
                image_part = image_part.ravel()
                image_part = image_part.reshape((1, cls.params["feature_len"]))
                dis_mat[count, :] = image_part
                count += 1
        decision = classifier.decision_function(dis_mat)
        return decision.reshape((image_shape[0] - patch_size + 1, image_shape[1] - patch_size + 1))

    @classmethod
    def batch_gen_dis_map(cls, image_feature_map):
        patch_size = cls.params["patch_size"] / cls.params["bins"]
        image_shape = image_feature_map.shape
        res = np.zeros((len(cls.classifiers), image_shape[0] - patch_size + 1, image_shape[1] - patch_size + 1),
                       dtype="float")
        for i in range(len(cls.classifiers)):
            temp = cls.gen_dis_map(image_feature_map, cls.classifiers[i])
            res[i, :, :] = temp
        return res

    @classmethod
    def dis_detector(cls, dis_tensor, proposal, axis=0):
        if axis != 0:
            proposal[0], proposal[1] = proposal[1], proposal[0]
            proposal[2], proposal[3] = proposal[3], proposal[2]
        proposal = proposal / cls.params["bins"]
        if proposal[0] == proposal[2]:
            proposal[2] += 1
        if proposal[1] == proposal[3]:
            proposal[3] += 1
        res = [0] * len(cls.classifiers)
        for i in range(len(cls.classifiers)):
            dis_mat = dis_tensor[i, proposal[0]:proposal[2], proposal[1]:proposal[3]]
            res[i] = np.amax(dis_mat)
        return res


