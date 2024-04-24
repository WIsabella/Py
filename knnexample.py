import numpy as np #数值计算库
import matplotlib.pyplot as plt  #绘图库
import pandas as pd  #数据分析与处理库

url = "D:\\iris.csv"  #定义一个变量url，包含数据集文件Iris数据集的路径

#Assign colum names to the dataset
#定义names列表，包含数据集各列名称
names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

#Read dataset to pandas dataframe
#使用pandas库的read_csv函数从指定路径读取CSV文件，并将列名设置为已定义的names
dataset =pd.read_csv(url,names=names)
print(dataset.head())

#将数据集分为特征矩阵x，和标签向量y。（x的属性决定着y的标识）iloc是用来按位置索引读取数据。
#.values将所选取的数据转化为numpy数组
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

#用于随机划分训练集和测试集的的函数train_test_split，测试集大小为原始的20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

#使用StandardScaler类来标准化数据，将数据缩放到具有零均值和单位方差。？？
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#自己定义一个简单的，无比重的KNN类
class KNN:
    def __init__(self, k=5):#构造函数，可以接受一个参数 self表示类自己，k=5表示默认为5
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):#用来训练模型的，它将训练数据X_train和标签y_train存储在类的变量实例里
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):#预测测试数据的标签。遍历每一个样本，调用_predict来预测，并将预测结果存在y_pred中，
        y_pred = [self._predict(x) for x in X_test]
        return np.array(y_pred)

    def _predict(self, x):#核心算法，预测单个样本的标签
        # 计算距离
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        # 获取距离最近的k个样本的索引
        k_indices = np.argsort(distances)[:self.k]
        # 对这k个样本的标签进行投票
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        from collections import Counter  # 计数库
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
if __name__ == "__main__":#程序主入口
    # X_train和y_train是已经准备好的训练数据
    # X_test是待预测的测试数据
    knn = KNN(k=5)#创建一个knn实例，
    knn.fit(X_train, y_train)#进行训练，
    y_pred = knn.predict(X_test)#进行预测
    print(y_pred)#打印结果

#from sklearn.neighbors import KNeighborsClassifier
#classifier =KNeighborsClassifier(n_neighbors=5)
#classifier.fit(X_train,y_train)
#y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test, y_pred))#打印混淆矩阵
print(classification_report(y_test, y_pred))#打印详细的分类报告，精确度、召回率及F1分数

error=[]#空列表，存储不同k值下的平均错误率
# Calculating error for K values between 1 and 40
for i in range(1, 40):#计算1-39平均错误率，对于每个k创建一个knn实例，训练，预测，
    knn = KNN(i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))#计算错误的标签率，添加到error里

#
plt.figure(figsize=(12, 6))#创建图形化窗口
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)#绘制错误率随k值变化的折线图
plt.title('Error Rate K Value')#图标标题
plt.xlabel('K Value')#图表x轴标签
plt.ylabel('Mean Error')#图表y轴标签
plt.show()
