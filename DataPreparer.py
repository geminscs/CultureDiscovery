import scipy.io as sio
import numpy as np
from FeatureExtractor import FeatureExtractor
from ProposalSizeFilter import ProposalSizeFilter
from DiscriminativeDetector import DiscriminativeDetector
from skimage.io import imread


def prepare_proposals():
    proposal_path = "../mid-data/raw_proposals.mat"
    image_base_path = "../images/kyoto/"

    mat_cont = sio.loadmat(proposal_path)
    proposals = mat_cont["res"]
    proposals = proposals[0, :]
    ProposalSizeFilter.initialize(80, 8000, 20)
    images = []
    for i in range(500):
        img = imread(image_base_path + str(i) + ".jpg")
        images.append(img)
    proposals = ProposalSizeFilter.filter_with_gradient(proposals, images)

    np.save("../mid-data/proposals.npy", proposals)


def prepare_feature():
    model_path = "/home/ge/tests/vgg16_weights.h5"
    proposal_path = "../mid-data/proposals.npy"
    image_base_path = "../images/kyoto/"
    feature_path = "../mid-data/feature/"

    proposals = np.load(proposal_path)
    FeatureExtractor.initialize(model_path)
    for i in range(10, 500):
        img = imread(image_base_path + str(i) + ".jpg")
        feature = FeatureExtractor.iterate_feature(proposals[i], img, axis=1)
        np.save(feature_path + str(i) + ".npy", feature)


def prepare_discrimination():
    class_path1 = "../mid-data/clfs_1.pkl"
    class_path2 = "../mid-data/clfs_2.pkl"
    discrimination_path = "../mid-data/dis2/"
    image_base_path = "../images/kyoto/"
    proposal_path = "../mid-data/proposals.npy"

    DiscriminativeDetector.initialize()
    DiscriminativeDetector.add_classifier(class_path1)
    DiscriminativeDetector.add_classifier(class_path2)

    proposals = np.load(proposal_path)
    for i in range(10, 500):
        img = imread(image_base_path + str(i) + ".jpg")
        img = DiscriminativeDetector.hog_feature(img)
        tensor = DiscriminativeDetector.batch_gen_dis_map(img)
        dis_mat = DiscriminativeDetector.batch_dis_detector(tensor, proposals[i], axis=1)
        np.save(discrimination_path + str(i) + ".npy", dis_mat)


def prepare_full_feature():
    image_base_path = "../images/kyoto/"
    model_path = "/home/ge/tests/vgg16_weights.h5"
    FeatureExtractor.initialize(model_path)
    images = np.zeros((500, 4096))
    for i in range(500):
        img = imread(image_base_path + str(i) + ".jpg")
        feature = FeatureExtractor.feature(img)
        images[i, :] = feature

    np.save("../mid-data/full_feature.npy", images)


if __name__ == "__main__":
    prepare_discrimination()


