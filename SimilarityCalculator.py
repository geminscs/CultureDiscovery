__author__ = 'admin'
from FeatureExtractor import FeatureExtractor
from DiscriminativeDetector import DiscriminativeDetector
from scipy.spatial import distance
import numpy as np


class SimilarityCalculator(object):
    @classmethod
    def region_similarity(cls, img1, img2, proposal1, proposal2):
        feature1 = FeatureExtractor.iterate_feature(proposal1, img1, axis=1)
        feature2 = FeatureExtractor.iterate_feature(proposal2, img2, axis=1)
        sim_mat = distance.cdist(feature1, feature2, "cosine")
        feature_map1 = DiscriminativeDetector.hog_feature(img1)
        feature_map2 = DiscriminativeDetector.hog_feature(img2)
        dis_tensor1 = DiscriminativeDetector.batch_gen_dis_map(feature_map1)
        dis_tensor2 = DiscriminativeDetector.batch_gen_dis_map(feature_map2)
        dis_mat1 = DiscriminativeDetector.batch_dis_detector(dis_tensor1, proposal1, axis=1)
        dis_mat2 = DiscriminativeDetector.batch_dis_detector(dis_tensor2, proposal2, axis=1)
        dis_mat = np.dot(dis_mat1, dis_mat2.T)
        res = sim_mat * dis_mat
        return res
