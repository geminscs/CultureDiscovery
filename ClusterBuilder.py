from scipy.spatial import distance
import numpy as np


class ClusterBuilder(object):  # TODO Select a suitable cluster algorithm
    """
    A customized cluster algorithm

    """

    def __init__(self, params):
        self.params = params

    def fit(self, X):
        labels = []
        return labels

    def predict(self, x):
        label = 0
        return label


class KNN(object):

    """
    Find K nearest neighbours
    """
    distance_matrix = None

    @classmethod
    def initialize(cls, mat_path):
        cls.distance_matrix = np.load(mat_path)

    @classmethod
    def set_distance_mat(cls, mat):
        cls.distance_matrix = mat

    @classmethod
    def knn(cls, target, k):
        """
        Find k nearest neighbours
        :param target: Target data point index
        :param k: Number of nearest neighbours
        :return: Index of k nearest neighbours
        """
        target_mat = cls.distance_matrix[target, :]
        target_mat = target_mat.reshape((1, 4096))
        dis_mat = distance.cdist(target_mat, cls.distance_matrix, "cosine")
        val = np.argsort(dis_mat)
        return val[0][1:k + 1]


