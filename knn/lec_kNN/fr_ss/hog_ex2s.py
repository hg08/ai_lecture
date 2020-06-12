# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 23:54:14 2018

@author: huang
"""
import matplotlib.pyplot as plt
from skimage import color
from skimage.feature import hog
from skimage import  exposure
from skimage import io


image = io.imread('zg.jpg')
image = color.rgb2gray(image)


# fd: HOG as a 1-D array
# hog_image: HOG for visualization
fd, hog_image = hog(image, orientations=4, pixels_per_cell=(8, 8),cells_per_block=(3, 3), visualise=True)
#fd, hog_image = hog(image,visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG')
plt.show()
