__author__ = 'admin'
__author__ = 'admin'
import scipy.io as sio
import numpy as np
from ProposalSizeFilter import ProposalSizeFilter
from skimage.io import imread
from DiscriminativeDetector import  DiscriminativeDetector
import pickle
from skimage.io import imsave
from scipy.spatial import distance

classifiers = []


def add_classifier(path):
    f = open(path, "rb")
    _classifier = pickle.load(f)
    f.close()
    for i in _classifier:
        if i != 0:
            classifiers.append(i)


def dis_detector(_dis_tensor, _proposal, axis=0):
    _proposal = _proposal / 8
    if _proposal[0] == _proposal[2]:
        _proposal[2] += 1
    if _proposal[1] == _proposal[3]:
        _proposal[3] += 1
    _res = [0] * 181
    for i in range(181):
        if axis == 0:
            _dis_mat = _dis_tensor[i, _proposal[0]:_proposal[2], _proposal[1]:_proposal[3]]
        else:
            _dis_mat = _dis_tensor[i, _proposal[1]:_proposal[3], _proposal[0]:_proposal[2]]
        _res[i] = np.amax(_dis_mat)
        #res[i] = np.sum(dis_mat) / (dis_mat.shape[0] * dis_mat.shape[1])
        _res[i] = 1 / (1 + np.exp(-5 * _res[i]))
        if _res[i] < 0.01:
            _res[i] = 0
        """elif res[i] < -1:
            res[i] = -1"""

    _res = np.array(_res)
    index = np.argsort(_res)
    _res[index[0:- 5]] = 0
    return _res


f = open("/Users/admin/Desktop/NewExp/dis_tensor1.pkl", "rb")
dis_tensor1 = pickle.load(f)
f.close()
f = open("/Users/admin/Desktop/NewExp/dis_tensor2.pkl", "rb")
dis_tensor2 = pickle.load(f)
f.close()

dis_mat1 = dis_detector(dis_tensor2, np.array([1, 106, 240, 333]), axis=1)
dis_mat2 = dis_detector(dis_tensor2, np.array([104, 208, 192, 329]), axis=1)

print dis_mat1


