import numpy as np
import bisect


def refine(proposals, axis=0):
    proposals -= 1
    _max = np.amax(proposals, axis=0)
    _min = np.amin(proposals, axis=0)
    if axis == 0:
        x1 = _min[0]
        x2 = _max[2]
        y1 = _min[1]
        y2 = _max[3]
    else:
        x1 = _min[1]
        x2 = _max[3]
        y1 = _min[0]
        y2 = _max[2]

    x_ratio = 1
    y_ratio = 1
    thre = 100.0
    if x2 - x1 > thre:
        x_ratio = thre / (x2 - x1)
    if y2 - y1 > thre:
        y_ratio = thre / (y2 - y1)
    k = 0
    for p in proposals:
        k += (p[2] - p[0]) * (p[3] - p[1]) * x_ratio * y_ratio
    k *= 0.80
    mat = np.zeros((int((x2 - x1) * x_ratio) + 1, int((y2 - y1) * y_ratio) + 1))
    for i in range(proposals.shape[0]):
        p = [0, 0, 0, 0]
        proposal = proposals[i]
        if axis == 0:
            p[0] = (proposal[0] - x1) * x_ratio
            p[1] = (proposal[1] - y1) * y_ratio
            p[2] = (proposal[2] - x1) * x_ratio
            p[3] = (proposal[3] - y1) * y_ratio
        else:
            p[0] = (proposal[1] - x1) * x_ratio
            p[1] = (proposal[0] - y1) * y_ratio
            p[2] = (proposal[3] - x1) * x_ratio
            p[3] = (proposal[2] - y1) * y_ratio
        a1,b1,a2,b2 = int(p[0]),int(p[1]),int(p[2]),int(p[3])

        mat[a1:a2, b1:b2] += 1
    res = maxSumSubmatrix(mat, k)
    res[0] = int(res[0] / x_ratio + x1)
    res[1] = int(res[1] / y_ratio + y1)
    res[2] = int(res[2] / x_ratio + x1)
    res[3] = int(res[3] / y_ratio + y1)
    return res


def maxSumSubmatrix(matrix, k):

    def maxSumSublist(vals, x1, x2):
        prefixSum = 0
        prefixSums = [float('inf')]
        for ind, val in enumerate(vals):
            bisect.insort(prefixSums, prefixSum)
            prefixSum += val
            if prefixSum < k:
                continue
            i = bisect.bisect_left(prefixSums, prefixSum - k)
            if i <= 0:
                continue
            if prefixSum - prefixSums[i - 1] >= k:
                if (r[2] - r[0]) * (r[3] - r[1]) > (x2 - x1) * (ind - i + 1):
                    r[0], r[1], r[2], r[3] = i - 1, x1, ind, x2

    columns = zip(*matrix)
    r = [0, 0, matrix.shape[0], matrix.shape[1]]
    for left in range(len(columns)):
        rowSums = np.array([0] * len(matrix))
        for ind, column in enumerate(columns[left:]):
            rowSums += np.asarray(column, dtype=int)
            maxSumSublist(rowSums, left, ind + left)

    return r

if __name__ == "__main__":
    p = np.load("/Users/admin/Desktop/pros_1.npy")
    p = p[0:10]
    r = [123, 39, 330, 340]
    r = refine(p, axis=1)
    print r