import numpy as np
from skimage.io import imread, imsave
from ClusterBuilder import KNN
from scipy.spatial import distance
import Refiner
from FeatureExtractor import FeatureExtractor

# Global Parameter Initialization
dataset_size = 500
feature_length = 4096


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
    return p_res


def main_iterate():
    res_box = []
    topn_box = []
    for i in range(dataset_size):
        source = KNN.knn(i, 5)
        pros = []
        for j in range(5):
            t_pros = match_res(i, source[j])
            if len(t_pros) >= 5 - j:
                pros.extend(t_pros[:5 - j])
            else:
                pros.extend(t_pros)
        if len(pros) == 0:
            res_box.append([])
            topn_box.append([])
            continue
        topn = 15
        if topn > len(pros):
            topn = len(pros)
        pros = np.array(pros)
        r = Refiner.refine(pros[:topn], axis=1)
        res_box.append(r)
        topn_box.append(pros[:topn])
        print i
    return [res_box, topn_box]

if __name__ == "__main__":
    refined_proposals = []
    test_image_path = "../test_images/kyoto2/"
    image_base_path = "../images/kyoto/"
    model_path = "/home/ge/tests/vgg16_weights.h5"
    FeatureExtractor.initialize(model_path)
    for iter in range(1):
        if iter == 0:
            KNN.initialize("../mid-data/full_feature.npy")
        else:
            distance_mat = np.zeros((dataset_size, feature_length))
            for i in range(dataset_size):
                image = imread(image_base_path + str(i) + ".jpg")
                if len(refined_proposals[0][i]) != 0:
                    t_refined_box = refined_proposals[0][i]
                    image = image[t_refined_box[0]:t_refined_box[2], t_refined_box[1]:t_refined_box[3]]
                    distance_mat[i, :] = FeatureExtractor.feature(image)
            KNN.set_distance_mat(distance_mat)
        refined_proposals = main_iterate()
        print str(iter) + " th iteration finished"
    refined_box = refined_proposals[0]
    candidate_box = refined_proposals[1]
    np.save("bbox.npy", refined_proposals)

    for i in range(dataset_size):
        image = imread(image_base_path + str(i) + ".jpg")
        t_refined_box = refined_box[i]
        if len(t_refined_box) == 0:
            continue
        for j in range(len(candidate_box[i])):
            proposal = candidate_box[i][j]
            if proposal[2] >= image.shape[1]:
                proposal[2] = image.shape[1] - 1
            if proposal[3] >= image.shape[0]:
                proposal[3] = image.shape[0] - 1
            image[proposal[1]:proposal[3], proposal[0], :] = [0, 255, 0]
            image[proposal[1]:proposal[3], proposal[2], :] = [0, 255, 0]
            image[proposal[1], proposal[0]:proposal[2], :] = [0, 255, 0]
            image[proposal[3], proposal[0]:proposal[2], :] = [0, 255, 0]
        image[t_refined_box[0]:t_refined_box[2], t_refined_box[1]:t_refined_box[1]+5, :] = [255, 0, 0]
        image[t_refined_box[0]:t_refined_box[2], t_refined_box[3]-5:t_refined_box[3], :] = [255, 0, 0]
        image[t_refined_box[0]:t_refined_box[0]+5, t_refined_box[1]:t_refined_box[3], :] = [255, 0, 0]
        image[t_refined_box[2]-5:t_refined_box[2], t_refined_box[1]:t_refined_box[3], :] = [255, 0, 0]
        imsave(test_image_path + str(i) + ".jpg", image)


