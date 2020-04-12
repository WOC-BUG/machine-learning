from load_data import load_CIFAR10      #读取数据
import numpy as np
import matplotlib.pyplot as plt

########################################1. 文件的读取 ########################################

cifar10_dir='cifar-10-batches-py'   #定义文件夹路径

#清空定义过的变量，若没定义过就不做处理
try:
    del X_train,Y_train
    del X_test,Y_test
    print('成功清除导入过的数据。')
except:     #try不成功时的输出错误信息
    pass    #不做任何事，用于占位语句

X_train,Y_train,X_test,Y_test=load_CIFAR10(cifar10_dir) #导入数据，X存RGB三色，Y存标签


# X的大小为N*W*H*3，Y的大小为图片的标签个数
#N：样本个数，W:宽，H:高，3：RGB三色
print("训练集，测试集：",X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
print("标签种类：",np.unique(Y_train))   #查看标签种类，一共有10种类别：airplane，automobile，bird，cat，deer，dog，frog，horse，ship，truck

classes=['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
num_classes = len(classes)  #样本种类的个数
samples_per_class=5     #每一个随机选择5个样本

#####从训练集中抽取样本，每个类别抽取samples_per_class个，并展示
vis=[0,0,0,0,0,0,0,0,0,0]   #记录每个类别的图片显示了多少张，达到十个就不再显示这个类别
plt.figure()    #创建图像窗口
cnt=1
for i in range(1,10000):
    vis[Y_train[i]]=vis[Y_train[i]]+1
    if vis[Y_train[i]]>=samples_per_class:
        continue
    else:
        plt.subplot(10,5,cnt)  #放在一个窗口里，10行5列
        cnt=cnt+1
        plt.imshow((X_train[i]*255).astype(np.uint8))
plt.show()

####统计并展示每个类别出现的次数
for i in range(0,9):
    print(classes[i]+'的样本的个数是：',end='')
    print(vis[i])

#####随机采样训练样本5000个，测试样本500个
num_training=5000
num_test=500

#生成随机数，取索引
random_training=np.random.permutation(len(X_train))     #训练集长度范围的随机数
training_index=random_training[:5000]   #训练集索引取前5000个
random_test=np.random.permutation(len(X_test))      #测试集长度范围的随机数
test_index=random_test[:500]  #测试集索引前500个

X_train=X_train[training_index]
Y_train=Y_train[training_index]

X_test=X_test[test_index]
Y_test=Y_test[test_index]

print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

######################################2.使用KNN算法识别图像######################################



###############################3.抽取图像特征，再用KNN算法来识别图片###############################


#################################4.使用PCA对图片做降维，并做可视化#################################