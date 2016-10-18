__author__ = 'admin'
import scipy.io as sio
import numpy as np
from FeatureExtractor import FeatureExtractor
from ProposalSizeFilter import ProposalSizeFilter
from skimage.io import imread
from scipy.spatial import distance


proposal_path = "/Users/admin/Desktop/NewExp/proposals.mat"
model_path = "/Users/admin/Documents/clothingTestImages/vgg16_weights.h5"
image_base_path = "/Users/admin/Desktop/NewExp/"

mat_cont = sio.loadmat(proposal_path)
proposals = mat_cont["res"]
proposals = proposals[0, :]
ProposalSizeFilter.initialize(80, 8000)
proposals = ProposalSizeFilter.filter(proposals)

FeatureExtractor.initialize(model_path)
img1 = imread(image_base_path + "1.jpg")
img2 = imread(image_base_path + "2.jpg")
feature1 = FeatureExtractor.batch_feature(proposals[0][0:2, :], img1, axis=1)
feature2 = FeatureExtractor.batch_feature(proposals[1][0:2, :], img2, axis=1)
dis_mat = distance.cdist(feature1, feature2)
img1_val = np.amin(dis_mat, axis=1)
img2_val = np.amin(dis_mat, axis=0)
print 1
#visualize here

