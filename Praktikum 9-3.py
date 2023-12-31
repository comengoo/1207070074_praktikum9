#import library
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from skimage import data 

#import image
img = cv2.imread ("paris.jpg", 0)

img_canny = cv2.Canny(img,100,200)

#fiter canny
fig, axes = plt.subplots(2, 2, figsize=(20,20))
ax = axes.ravel()

ax[0].imshow(img,cmap ='gray')
ax[0].set_title("Citra Input")
ax[1].hist(img.ravel(), bins = 256)
ax[1].set_title("Histogram Citra Input")

ax[2].imshow(img_canny,cmap ='gray')
ax[2].set_title("Citra Input")
ax[3].hist(img_canny.ravel(), bins = 256)
ax[3].set_title("Histogram Citra Input")

plt.show()