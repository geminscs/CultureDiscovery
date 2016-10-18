__author__ = 'admin'
import numpy as np
import scipy.io as sio
#from skimage.exposure import is_low_contrast


class ProposalSizeFilter(object):
    lengthThres = 0
    AreaThres = 0

    @classmethod
    def initialize(cls, lt, at):
        cls.lengthThres = lt
        cls.AreaThres = at

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


if __name__ == "__main__":
    mat_content = sio.loadmat("/Users/admin/Desktop/proposals.mat")
    proposals = mat_content["res"]
    proposals = proposals[0, :]
    ProposalSizeFilter.initialize(80, 8000)
    proposals = ProposalSizeFilter.filter(proposals)
    print proposals





