__author__ = 'admin'
import pickle
from skimage.feature import hog
from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np


class DiscriminativeDetector(object):
    classifiers = []
    classifier_weight = []
    params = dict()

    @classmethod
    def initialize(cls, patch_size=80, bins=8, feature_len=800):
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
    def set_weight(cls, weight):
        cls.classifier_weight = weight

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

        proposal = proposal / cls.params["bins"]
        if proposal[0] == proposal[2]:
            proposal[2] += 1
        if proposal[1] == proposal[3]:
            proposal[3] += 1
        res = [0] * len(cls.classifiers)
        for i in range(len(cls.classifiers)):
            if axis == 0:
                dis_mat = dis_tensor[i, proposal[0]:proposal[2], proposal[1]:proposal[3]]
            else:
                dis_mat = dis_tensor[i, proposal[1]:proposal[3], proposal[0]:proposal[2]]
            #res[i] = np.amax(dis_mat)
            res[i] = np.sum(dis_mat) / (dis_mat.shape[0] * dis_mat.shape[1])
            res[i] = 1 / (1 + np.exp(-1 * res[i]))
            if res[i] < 0.01:
                res[i] = 0
            """elif res[i] < -1:
                res[i] = -1"""

        res = np.array(res)
        index = np.argsort(res)
        res[index[0:- 5]] = 0
        if sum(res) > 0:
            res = res / sum(res)
        #res = np.amax(res)
        if len(cls.classifier_weight) != 0:
            res = res * cls.classifier_weight
        return res

    @classmethod
    def batch_dis_detector(cls, dis_tensor, proposals, axis=0):
        res = np.zeros((proposals.shape[0], len(cls.classifiers)))
        for i in range(proposals.shape[0]):
            proposal = proposals[i]
            dis_vec = cls.dis_detector(dis_tensor, proposal, axis=axis)
            res[i, :] = dis_vec
        return res

    @classmethod
    def batch_dis_detector_single(cls, dis_tensor, proposals, axis=0):
        res = []
        for i in range(proposals.shape[0]):
            proposal = proposals[i]
            dis_vec = cls.dis_detector(dis_tensor, proposal, axis=axis)
            res.append(dis_vec)
        return np.array(res).reshape((1, len(res)))

    @classmethod
    def get_weight(cls, path_p, path_n):
        f = open(path_p, "rb")
        fp = pickle.load(f)
        f.close()
        f = open(path_n, "rb")
        fn = pickle.load(f)
        f.close()
        fp_mat = []
        for i in range(len(fp)):
            fp_mat.append(fp[i]["feature"])
        fp_mat = np.array(fp_mat)
        fn_mat = []
        for i in range(len(fn)):
            fn_mat.append(fn[i]["feature"])
        fn_mat = np.array(fn_mat)
        weight = [0] * len(cls.classifiers)
        for i in range(len(weight)):
            confidence_p = cls.classifiers[i].decision_function(fp_mat)
            fire_p = len(confidence_p[confidence_p >= 0]) + 1
            confidence_n = cls.classifiers[i].decision_function(fn_mat)
            fire_n = len(confidence_n[confidence_n >= 0]) + 1
            weight[i] = 1 / float(fire_n)
        cls.classifier_weight = weight

    @classmethod
    def hog_feature(cls, image):
        if image.ndim > 1:
            image = rgb2gray(image)
        f = hog(image, orientations=8, pixels_per_cell=(cls.params["bins"],
                                                           cls.params["bins"]), cells_per_block=(1, 1))
        return f.reshape((image.shape[0] / cls.params["bins"], image.shape[1] / cls.params["bins"], cls.params["bins"]))

if __name__ == "__main__":
    """DiscriminativeDetector.initialize()
    DiscriminativeDetector.add_classifier("/Users/admin/Desktop/Patch/20160613_kyoto/clfs_1.pkl")
    DiscriminativeDetector.add_classifier("/Users/admin/Desktop/Patch/20160613_kyoto/clfs_2.pkl")
    image = imread("/Users/admin/Desktop/NewExp/1.jpg", as_grey=True)
    feature_map = DiscriminativeDetector.hog_feature(image)
    dis_tensor = DiscriminativeDetector.batch_gen_dis_map(feature_map)
    dis_mat = DiscriminativeDetector.batch_dis_detector(dis_tensor, np.array([[1, 1, 100, 100], [1, 2, 100, 101]]))
    print dis_tensor.shape"""
    DiscriminativeDetector.initialize()
    DiscriminativeDetector.add_classifier("/Users/admin/Desktop/Patch/20160613_kyoto/clfs_1.pkl")
    DiscriminativeDetector.add_classifier("/Users/admin/Desktop/Patch/20160613_kyoto/clfs_2.pkl")
    DiscriminativeDetector.get_weight("/Users/admin/Desktop/kyoto2_feature.pkl", "/Users/admin/Desktop/nature_feature.pkl")
    f = open("/Users/admin/Desktop/NewExp/weights2.pkl", "wb")
    pickle.dump(DiscriminativeDetector.classifier_weight, f)
    f.close()
