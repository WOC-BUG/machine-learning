import matplotlib.pyplot as plt
import numpy as np
from itertools import product #可视化模块
from sklearn.neighbors import KNeighborsClassifier

# 生成一些随机样本
n_points=100
x1=np.random.multivariate_normal([1,50],[[1,0],[0,10]],n_points) # 多元正态分布矩阵
x2=np.random.multivariate_normal([2,50],[[1,0],[0,10]],n_points) # 多元正态分布矩阵
X=np.concatenate([x1,x2]) # 连接两个矩阵形成X
Y=np.array([0]*n_points+[1]*n_points) # 100个0，100个1
print(X.shape,Y.shape)

# KNN模型的训练过程
clfs=[]
neighbors=[1,3,5,9,11,13,15,17,19]
for i in range(len(neighbors)): #i从0到len(neighbors)-1
    clfs.append(KNeighborsClassifier(n_neighbors=neighbors[i]).fit(X,Y))

# 可视化结果
x_min,x_max=X[:,0].min()-1,X[:,0].max()+1 # X[:,0]表示全部数据的第0个数据
y_min,y_max=X[:,1].min()-1,X[:,1].max()+1 # X[:,1]表示全部数据的第1个数据
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1)) #生成网格点坐标矩阵

f,axarr=plt.subplots(3,3,sharex='col',sharey='row',figsize=(15,12)) # 把父图分解成3*3的多个子图
for idx,clf,tt in zip(product([0,1,2],[0,1,2]),
                      clfs,
                      ['KNN(k=%d)'%k for k in neighbors]):
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()]) # np.c_[]按列连接两个矩阵,ravel()将多维数组转换为一维数组
    Z=Z.reshape(xx.shape)

    axarr[idx[0],idx[1]].contourf(xx,yy,Z,alpha=0.4) # 画等高线图
    axarr[idx[0],idx[1]].scatter(X[:,0],X[:,1],c=Y,s=20,edgecolor='k') # 画点
    axarr[idx[0],idx[1]].set_title(tt)

plt.show()