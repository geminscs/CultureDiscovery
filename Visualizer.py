__author__ = 'admin'
import numpy as np
import pickle
from skimage.io import imread, imsave
from ProposalSizeFilter import ProposalSizeFilter
import scipy.io as sio

proposal_path = "/Users/admin/Desktop/NewExp/proposals.mat"
mat_cont = sio.loadmat(proposal_path)
proposals = mat_cont["res"]
proposals = proposals[0, :]
ProposalSizeFilter.initialize(80, 8000)
proposals = ProposalSizeFilter.filter(proposals)
p2 = proposals[1]

f = open("/Users/admin/Desktop/NewExp/dis_mat_cosine.pkl", "rb")
dis_mat = pickle.load(f)
img2_val = np.amin(dis_mat, axis=0)
img2_index = np.argsort(img2_val)
image = imread("/Users/admin/Desktop/NewExp/2.jpg")
for i in range(5):
    proposal = p2[img2_index[i], :]
    if proposal[2] >= image.shape[1]:
        proposal[2] = image.shape[1] - 1
    if proposal[3] >= image.shape[0]:
        proposal[3] = image.shape[0] - 1
    image[proposal[1]:proposal[3], proposal[0], :] = [0, 255, 0]
    image[proposal[1]:proposal[3], proposal[2], :] = [0, 255, 0]
    image[proposal[1], proposal[0]:proposal[2], :] = [0, 255, 0]
    image[proposal[3], proposal[0]:proposal[2], :] = [0, 255, 0]

imsave("/Users/admin/Desktop/NewExp/5_5.jpg", image)
