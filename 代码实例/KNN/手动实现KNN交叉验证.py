# 手动实现KNN交叉验证
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold # 用于K折交叉验证


# 导入iris数据集
iris=datasets.load_iris()
x=iris.data
y=iris.target
print(x.shape,y.shape)

# 定义想要搜索的K值
ks=[1,3,5,7,9,11,13,15]

# 进行5折交叉验证，KFold返回的是每一折中训练数据和验证数据的index
# 假设数据样本为：[1,3,5,6,11,12,43,12,44,2]共10个
# 则返回的KFold格式为（前面为训练数据，后面是验证集）
# [0,1,3,5,6,7,8,9],[2,4]
# [0,1,2,4,6,7,8,9],[3,5]
# [1,2,3,4,5,6,7,8],[0,9]
# [1,2,3,4,5,6,7,8],[0,9]
# [0,2,3,4,5,6,8,9],[1,7]
kf=KFold(n_splits=5,shuffle=True,random_state=2001)

# 定义变量，用保存当前最好的K值和对应的准确率
best_k=ks[0]
best_score=0

# 循环每一个k值
for k in ks:
    curr_score=0
    for train_index,valid_index in kf.split(x):
        # 每一折的训练以及计算准确率
        clf=KNeighborsClassifier(n_neighbors=k)
        clf.fit(x[train_index],y[train_index])
        curr_score=curr_score+clf.score(x[valid_index],y[valid_index])
    # 求5折训练的平均准确率
    avg_score=curr_score/5
    if(avg_score>best_score):
        best_k=k
        best_score=avg_score
    print("current best score is: %.2f"%best_score,"best k: %d"%best_k)

print("after cross validation,the final best k is: %d"%best_k)