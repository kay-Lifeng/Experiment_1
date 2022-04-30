import matplotlib.pyplot as plt
import matplotlib.image as Image
import numpy as np

im1 = Image.imread("/Users/wanglifeng/Downloads/bird.png")
im2 = Image.imread("/Users/wanglifeng/Downloads/dog.png")
lam= 4*0.1
im_mixup1 = (im1*lam+im2*(1-lam))
plt.imshow(im_mixup)
plt.show()