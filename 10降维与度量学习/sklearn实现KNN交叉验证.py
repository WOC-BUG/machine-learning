# sklearn实现KNN交叉验证
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV # 通过网格方式来搜索参数

# 导入iris数据集
iris=datasets.load_iris()
x=iris.data
y=iris.target

# 设定想要搜索的K值,'n_neighbors'是sklearn中KNN的参数
parameters={'n_neighbors':[1,3,5,7,9,11,13,15]}
knn=KNeighborsClassifier() # 注意，这里不用指定参数

# 通过GridSearchCV来搜索最好的K值
# 该模块内部是对每一个K值进行了评估
clf=GridSearchCV(knn,parameters,cv=5)
clf.fit(x,y)

# 输出最好的参数以及准确率
print("best score is: %.2f"%clf.best_score_,"best k: ",clf.best_params_)