import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from skimage.io import imread, imshow

a = np.array([[1,2,3],[1,2,3]])
b = [1, 4, 9, 0, 0]
c = [1, 3]
del b[1]
print np.sum(a) / (a.shape[0] * a.shape[1])
