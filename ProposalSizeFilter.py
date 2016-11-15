__author__ = 'admin'
import numpy as np
import scipy.io as sio
#from skimage.exposure import is_low_contrast
from skimage.morphology import disk
from skimage.filters.rank import gradient
from skimage.color import rgb2gray
from skimage.io import imread

class ProposalSizeFilter(object):
    lengthThres = 0
    AreaThres = 0
    GradientThres = 0

    @classmethod
    def initialize(cls, lt, at, gt=5):
        cls.lengthThres = lt
        cls.AreaThres = at
        cls.GradientThres = gt

    @classmethod
    def filter(cls, proposals):
        res_proposals = []
        for image in proposals:
            s = image.shape
            image_proposal = np.zeros((1, 4))
            init = True
            for i in range(s[0]):
                width = image[i][2] - image[i][0]
                height = image[i][3] - image[i][1]
                if width > cls.lengthThres and height > cls.lengthThres:
                    sp = width * height
                    if sp > cls.AreaThres:
                        if init:
                            image_proposal[0, :] = image[i, :]
                            init = False
                        else:
                            image_proposal = np.r_[image_proposal, image[i, :].reshape((1, 4))]
            res_proposals.append(image_proposal)
        return res_proposals

    @classmethod
    def filter_with_gradient(cls, proposals, images, axis=0):
        res_proposals = []
        for j in range(proposals.shape[0]):
            proposal = proposals[j]
            image = images[j]
            image = rgb2gray(image)
            image = gradient(image, disk(1))
            s = proposal.shape
            image_proposal = np.zeros((1, 4))
            init = True
            for i in range(s[0]):
                width = proposal[i][2] - proposal[i][0]
                height = proposal[i][3] - proposal[i][1]
                if width > cls.lengthThres and height > cls.lengthThres:
                    sp = width * height
                    if sp > cls.AreaThres:
                        gra_sum = 0
                        if axis == 0:
                            gra_sum = np.sum(image[proposal[i][0]:proposal[i][2], proposal[i][1]:proposal[i][3]])
                        else:
                            gra_sum = np.sum(image[proposal[i][1]:proposal[i][3], proposal[i][0]:proposal[i][2]])
                        if gra_sum / sp < cls.GradientThres:
                            continue
                        if init:
                            image_proposal[0, :] = proposal[i, :]
                            init = False
                        else:
                            image_proposal = np.r_[image_proposal, proposal[i, :].reshape((1, 4))]
            res_proposals.append(image_proposal)
        return res_proposals


if __name__ == "__main__":
    mat_content = sio.loadmat("/Users/admin/Desktop/NewExp/proposals.mat")
    proposals = mat_content["res"]
    proposals = proposals[0, :]
    ProposalSizeFilter.initialize(80, 8000)
    images = []
    img = imread("/Users/admin/Desktop/NewExp/1.jpg")
    images.append(img)
    img = imread("/Users/admin/Desktop/NewExp/2.jpg")
    images.append(img)
    proposals = ProposalSizeFilter.filter_with_gradient(proposals, images, gra_thres=20,axis=1)
    print proposals





