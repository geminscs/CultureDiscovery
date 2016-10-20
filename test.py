__author__ = 'admin'
import scipy.io as sio
import numpy as np
from ProposalSizeFilter import ProposalSizeFilter
from skimage.io import imread
from DiscriminativeDetector import  DiscriminativeDetector
import pickle
from skimage.io import imsave
from scipy.spatial import distance


proposal_path = "/Users/admin/Desktop/NewExp/proposals.mat"
image_base_path = "/Users/admin/Desktop/NewExp/"
sim_path = "/Users/admin/Desktop/NewExp/dis_mat_cosine.pkl"
weight_path = "/Users/admin/Desktop/NewExp/weights.pkl"

mat_cont = sio.loadmat(proposal_path)
proposals = mat_cont["res"]
proposals = proposals[0, :]
ProposalSizeFilter.initialize(80, 8000)
proposals = ProposalSizeFilter.filter(proposals)
p2 = proposals[1]
img1 = imread(image_base_path + "1.jpg")
img2 = imread(image_base_path + "2.jpg")

f = open(sim_path, "rb")
sim_mat = pickle.load(f)
f.close()
sim_mat = 1 - sim_mat

f = open(weight_path, "rb")
weights = pickle.load(f)
weights = np.array(weights)
weights = weights / sum(weights)
f.close()

taboo = [17, 18, 27, 78, 82, 94, 122]


DiscriminativeDetector.initialize()
DiscriminativeDetector.add_classifier("/Users/admin/Desktop/Patch/20160613_kyoto/clfs_1.pkl")
DiscriminativeDetector.add_classifier("/Users/admin/Desktop/Patch/20160613_kyoto/clfs_2.pkl")
DiscriminativeDetector.set_weight(weights)

c = []
w = []
for i in range(len(DiscriminativeDetector.classifiers)):
    if i in taboo:
        continue
    else:
        c.append(DiscriminativeDetector.classifiers[i])
        w.append(DiscriminativeDetector.classifier_weight[i])
DiscriminativeDetector.classifiers = c
DiscriminativeDetector.classifier_weight = w

feature_map1 = DiscriminativeDetector.hog_feature(img1)
feature_map2 = DiscriminativeDetector.hog_feature(img2)
dis_tensor1 = DiscriminativeDetector.batch_gen_dis_map(feature_map1)
dis_tensor2 = DiscriminativeDetector.batch_gen_dis_map(feature_map2)
dis_mat1 = DiscriminativeDetector.batch_dis_detector(dis_tensor1, proposals[0], axis=1)
dis_mat2 = DiscriminativeDetector.batch_dis_detector(dis_tensor2, proposals[1], axis=1)
dis_mat = np.dot(dis_mat1, dis_mat2.T)
res = sim_mat * dis_mat
#res = dis_mat
img2_val = np.amax(res, axis=0)
img2_index = np.argsort(img2_val)[::-1]
image = imread("/Users/admin/Desktop/NewExp/2.jpg")

for i in range(10):
    proposal = p2[img2_index[i], :]
    if proposal[2] >= image.shape[1]:
        proposal[2] = image.shape[1] - 1
    if proposal[3] >= image.shape[0]:
        proposal[3] = image.shape[0] - 1
    image[proposal[1]:proposal[3], proposal[0], :] = [0, 255, 0]
    image[proposal[1]:proposal[3], proposal[2], :] = [0, 255, 0]
    image[proposal[1], proposal[0]:proposal[2], :] = [0, 255, 0]
    image[proposal[3], proposal[0]:proposal[2], :] = [0, 255, 0]

imsave("/Users/admin/Desktop/NewExp/6_20.jpg", image)

