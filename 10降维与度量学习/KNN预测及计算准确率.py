from sklearn import datasets #导入样本数据
from sklearn.model_selection import train_test_split #做数据集的分割，把数据分成训练集和测试集
from sklearn.neighbors import KNeighborsClassifier #导入KNN的模块
import numpy as np

iris=datasets.load_iris() #导入iris数据集，包含三个类别，适合分类问题
x=iris.data
y=iris.target
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=2003)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)
clf=KNeighborsClassifier(n_neighbors=3) #取k=3
clf.fit(x_train,y_train)

#预测及计算准确率
correct=np.count_nonzero((clf.predict(x_test)==y_test)==True) #比较每个x_test预测出来的值和y_test值，相等为1不等为0,计算1的个数
print("Accuarcy is: %.3f"%(correct/len(x_test)))