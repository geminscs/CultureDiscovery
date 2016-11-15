import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from skimage.io import imread, imshow, imsave
import scipy.io as sio
import pickle
from scipy.spatial import distance
import Refiner
from sklearn import svm

#trapSerial.py
#example to run: python trapSerial.py 0.0 1.0 10000

import numpy
import sys

#takes in command-line arguments [a,b,n]
a = 0
b = 1
n = 1000000

def f(x):
        return x*x

def integrateRange(a, b, n):
        '''Numerically integrate with the trapezoid rule on the interval from
        a to b with n trapezoids.
        '''
        integral = -(f(a) + f(b))/2.0
        # n+1 endpoints, but n trapazoids
        for x in numpy.linspace(a,b,n+1):
                integral = integral + f(x)
        integral = integral* (b-a)/n
        return integral

integral = integrateRange(a, b, n)
print "With n =", n, "trapezoids, our estimate of the integral\
from", a, "to", b, "is", integral

