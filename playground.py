import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from skimage.io import imread, imshow

'''points = np.arange(-5, 5, 0.01)
dx, dy = np.meshgrid(points, points)
z = (np.sin(dx)+np.sin(dy))
plt.imshow(z)
plt.colorbar()
plt.title('plot for sin(x)+sin(y)')
plt.show()'''

img = mpimg.imread('/Users/admin/Desktop/NewExp/2.jpg')
fig1 = plt.imshow(img)
ax1 = fig1.add_subplot(111, aspect='equal')
ax1.add_patch(
    patches.Rectangle(
        (0.1, 0.1),   # (x,y)
        0.5,          # width
        0.5,          # height
    )
)
fig1.savefig('/Users/admin/Desktop/NewExp/4.jpg', dpi=90, bbox_inches='tight')
