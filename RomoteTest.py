__author__ = 'admin'
import scipy.io as sio
import numpy as np
from FeatureExtractor import FeatureExtractor
from ProposalSizeFilter import ProposalSizeFilter
from skimage.io import imread
from scipy.spatial import distance
import pickle

proposal_path = "../proposals.mat"
model_path = "/home/ge/tests/vgg16_weights.h5"
image_base_path = "../"

mat_cont = sio.loadmat(proposal_path)
proposals = mat_cont["res"]
proposals = proposals[0, :]
ProposalSizeFilter.initialize(80, 8000)
proposals = ProposalSizeFilter.filter(proposals)

FeatureExtractor.initialize(model_path)
img1 = imread(image_base_path + "1.jpg")
img2 = imread(image_base_path + "2.jpg")
feature1 = FeatureExtractor.iterate_feature(proposals[0], img1, axis=1)
feature2 = FeatureExtractor.iterate_feature(proposals[1], img2, axis=1)
dis_mat = distance.cdist(feature1, feature2, "cosine")
f = open("dis_mat_cosine.pkl", "wb")
pickle.dump(dis_mat, f)
f.close()

