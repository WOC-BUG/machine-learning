# 常用的图像特征
# 1. 颜色特征(Color Histogram)
# 2. SIFT(scale-invariant feature transform)
# 3. HOG(Histogram of Oriented Gradient)

import matplotlib.pyplot as plt
import pylab

# 读取图片数据，存放到img
img=plt.imread('E:\图\girl.jpg')
print(img.shape) # 打印图片大小

plt.imshow(img) # 处理图像，并显示图像格式
plt.show() # 显示图像