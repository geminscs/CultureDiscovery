import scipy.io as sio
import numpy as np
from ProposalSizeFilter import ProposalSizeFilter
from skimage.io import imread
from DiscriminativeDetector import DiscriminativeDetector
import pickle
from skimage.io import imsave
from scipy.spatial import distance
import sys
import Refiner

def best_match(target):
    feature_path = "../mid-data/feature/"
    discrimination_path = "../mid-data/discrimination/"
    _rank = []
    target_feature = np.load(feature_path + str(target) + ".npy")
    target_dis = np.load(discrimination_path + str(target) + ".npy")
    for i in range(500):
        if i == target:
            _rank.append(0)
        else:
            i_feature = np.load(feature_path + str(i) + ".npy")
            i_dis = np.load(discrimination_path + str(i) + ".npy")
            sim_mat = distance.cdist(target_feature, i_feature, "cosine")
            sim_mat = 1 - sim_mat
            dis_mat = np.dot(target_dis, i_dis.T)
            res = sim_mat * dis_mat
            val = np.amax(res, axis=1)
            val = np.sort(val)[::-1]
            s = np.sum(val[0:10])
            _rank.append(s)
    return _rank


def match_res(i1, i2):
    feature_path = "../mid-data/feature/"
    discrimination_path = "../mid-data/dis2/"
    proposal_path = "../mid-data/proposals.npy"

    proposals = np.load(proposal_path)

    feature1 = np.load(feature_path + str(i1) + ".npy")
    dis1 = np.load(discrimination_path + str(i1) + ".npy")

    feature2 = np.load(feature_path + str(i2) + ".npy")
    dis2 = np.load(discrimination_path + str(i2) + ".npy")
    sim_mat = distance.cdist(feature1, feature2, "cosine")
    sim_mat = 2 - sim_mat
    dis_mat = np.dot(dis1, dis2.T)
    res = sim_mat * dis_mat
    _val = np.amax(res, axis=1)
    val = np.argsort(_val)[::-1]
    p = proposals[i1]
    p_res = []
    for i in range(20):
        if i >= len(val):
            break
        proposal = p[val[i], :]
        if _val[val[i]] <= 0.05:
            continue
        p_res.append(proposal)
    return np.array(p_res)


def match_res_local(i1, i2):
    feature_path = "/Users/admin/Desktop/NewExp/feature/"
    discrimination_path = "/Users/admin/Desktop/NewExp/discrimination/"
    proposal_path = "/Users/admin/Desktop/NewExp/proposals.npy"
    image_base_path = "/Users/admin/Desktop/Dataset/city/kyoto/"
    test_image_path = "/Users/admin/Desktop/NewExp/"

    proposals = np.load(proposal_path)

    feature1 = np.load(feature_path + str(i1) + ".npy")
    feature2 = np.load(feature_path + str(i2) + ".npy")

    img1 = imread(image_base_path + str(i1) + ".jpg")
    img2 = imread(image_base_path + str(i2) + ".jpg")
    dis_tensor1 = DiscriminativeDetector.batch_gen_dis_map(DiscriminativeDetector.hog_feature(img1))
    dis_tensor2 = DiscriminativeDetector.batch_gen_dis_map(DiscriminativeDetector.hog_feature(img2))
    dis1 = DiscriminativeDetector.batch_dis_detector(dis_tensor1, proposals[i1], axis=1)
    dis2 = DiscriminativeDetector.batch_dis_detector(dis_tensor2, proposals[i2], axis=1)
    sim_mat = distance.cdist(feature1, feature2, "cosine")
    sim_mat = 2 - sim_mat
    dis_mat = np.dot(dis1, dis2.T)
    res = sim_mat * dis_mat
    _val = np.amax(res, axis=1)
    val = np.argsort(_val)[::-1]
    v = np.sort(_val)[::-1]
    print np.sum(v[0:10])
    image = imread(image_base_path + str(i1) + ".jpg")
    p = proposals[i1]
    p_temp = []
    for i in range(20):
        if _val[val[i]] <= 0.1:
            continue
        if val[i] >= len(p):
            continue
        proposal = p[val[i], :]
        p_temp.append(proposal)
        if proposal[2] >= image.shape[1]:
            proposal[2] = image.shape[1] - 1
        if proposal[3] >= image.shape[0]:
            proposal[3] = image.shape[0] - 1
        image[proposal[1]:proposal[3], proposal[0], :] = [0, 255, 0]
        image[proposal[1]:proposal[3], proposal[2], :] = [0, 255, 0]
        image[proposal[1], proposal[0]:proposal[2], :] = [0, 255, 0]
        image[proposal[3], proposal[0]:proposal[2], :] = [0, 255, 0]
    print "match finished"
    p_temp = np.array(p_temp)
    np.save("/Users/admin/Desktop/NewExp/p_temp.npy", p_temp)
    #r = Refiner.refine(p_temp, axis=1)
    r = [39, 12, 340, 219]
    print r

    if r[2] >= image.shape[1]:
        r[2] = image.shape[1] - 1
    if r[3] >= image.shape[0]:
        r[3] = image.shape[0] - 1
    image[r[1]:r[3], r[0], :] = [255, 0, 0]
    image[r[1]:r[3], r[2], :] = [255, 0, 0]
    image[r[1], r[0]:r[2], :] = [255, 0, 0]
    image[r[3], r[0]:r[2], :] = [255, 0, 0]
    imsave(test_image_path + str(i1) + "_" + str(i2) + ".jpg", image)


def full_knn(target):
    full_feature_path = "../mid-data/full_feature.npy"
    mat = np.load(full_feature_path)
    target_mat = mat[target, :]
    target_mat = target_mat.reshape((1, 4096))
    dis_mat = distance.cdist(target_mat, mat, "cosine")
    val = np.argsort(dis_mat)
    return int(val[0][1])


def main_iterate_1():
    test_image_path = "../test_images/kyoto/"
    image_base_path = "../images/kyoto/"
    for i in range(346,500):
        source = full_knn(i)
        print i
        pros = match_res(i, source)
        if len(pros) == 0:
            continue
        topn = 10
        image = imread(image_base_path + str(i) + ".jpg")
        if topn > len(pros):
            topn = len(pros)
        r = Refiner.refine(pros[:topn], axis=1)
        for j in range(topn):
            proposal = pros[j]
            if proposal[2] >= image.shape[1]:
                proposal[2] = image.shape[1] - 1
            if proposal[3] >= image.shape[0]:
                proposal[3] = image.shape[0] - 1
            image[proposal[1]:proposal[3], proposal[0], :] = [0, 255, 0]
            image[proposal[1]:proposal[3], proposal[2], :] = [0, 255, 0]
            image[proposal[1], proposal[0]:proposal[2], :] = [0, 255, 0]
            image[proposal[3], proposal[0]:proposal[2], :] = [0, 255, 0]
        image[r[0]:r[2], r[1], :] = [255, 0, 0]
        image[r[0]:r[2], r[3], :] = [255, 0, 0]
        image[r[0], r[1]:r[3], :] = [255, 0, 0]
        image[r[2], r[1]:r[3], :] = [255, 0, 0]
        imsave(test_image_path + str(i) + ".jpg", image)


if __name__ == "__main__":
    main_iterate_1()

