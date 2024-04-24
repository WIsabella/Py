# nump数值计算，matplotlib绘图,
import numpy as np  # 数值计算
from matplotlib import pyplot as plt  # 绘图


def Edistance(feature, center):
    distance = 0
    for i in range(len(feature)):
        distance += pow((feature[i] - center[i]), 2)
    return distance ** 0.5

def compute_average(features):
    if len(features) == 0:
        return None  # 如果没有数据点，返回None
    # 初始化总和向量和计数器
    sum_vector = [0] * len(features[0])
    count = 0
    # 遍历所有特征向量，累加它们的值
    for feature in features:
        for i in range(len(feature)):
            sum_vector[i] += feature[i]
        count += 1
    # 计算平均值
    average = [sum_val / count for sum_val in sum_vector]
    return average

# 假设self.clf_是一个字典，其中包含了不同类别的数据点
# c是当前类别的标签
# c = 0  # 示例类别标签
# self.clf_ = {c: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}  # 示例数据点

# 计算类别c的所有数据点的平均值


class K_Means(object):
    # 构造函数：k聚类数量，tolerance中心点变化小于该值后停止，max_iter最大迭代次数
    def __init__(self, k=2, tolerance=0.0001, max_iter=300):
        self.k_ = k
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    # fit训练算法：
    def fit(self, data):
        # 初始化中心点：从数据中随机选取k个数据点作为初始中心
        self.centers_ = {i: data[i] for i in range(self.k_)}

        for i in range(self.max_iter_):  # 循环
            self.clf_ = {i: [] for i in range(self.k_)}  # 创建字典self.clf_存储每个聚类的数据点。字典的键是聚类的索引，值存储数据点。
            # 计算每个点到中心点的距离，并将数据点分配到最近的中心点所在的聚类中
            for feature in data:
                # distance = [np.linalg.norm(feature - self.centers_[center]) for center in self.centers_]
                distance = [Edistance(feature, self.centers_[center]) for center in self.centers_]
                # print (distance)
                classification = distance.index(min(distance))
                # print(classification)
                self.clf_[classification].append(feature)

            prev_centers = dict(self.centers_)  # 新字典存储上一次迭代的中心点
            # print(prev_centers)

            # 更新中心点，若每个聚类中，若非空，则计算所有数据点的平均值，并将平均值作为新的中心点。
            for c in self.clf_:
                if self.clf_[c]:  # 检查聚类是否为空
                    # self.centers_[c] = np.average(self.clf_[c], axis=0)
                    self.centers_[c] = compute_average(self.clf_[c])
            optimized = True  # 检查中心点是否收敛于tolerance
            # 如果任意一个中心点在迭代中发生了变化，就改为false并退出
            for center in self.centers_:
                org_centers = prev_centers[center]
                cur_centers = self.centers_[center]
                if not np.all(np.equal(cur_centers, org_centers)):
                    optimized = False
                    break

            if optimized:
                break

    def predict(self, p_data):
        # 计算预测点与中心点欧氏距离，存储在distance中，调用了np中计算欧氏距离的函数
        distance = [np.linalg.norm(p_data - self.centers_[center]) for center in self.centers_]
        # 找到最小距离的索引，对应聚类标签
        index = distance.index(min(distance))
        return index


if __name__ == '__main__':
    colors = ['blue', 'green', 'red', 'purple']  # 颜色
    markers = ['o', 'v', 's', '*']  # 图形
    # 数据组
    x1 = np.array(
        [39, 9, 11, 44, 14, 8, 25, 0, 12, 45, 30, 32, 45, 15, 15, 6, 6, 46, 39, 13, 47, 1, 9, 17, 38, 34, 25, 2, 37, 9])
    x2 = np.array(
        [19, 41, 43, 35, 6, 36, 4, 13, 34, 38, 13, 25, 44, 14, 44, 10, 6, 45, 2, 19, 48, 29, 49, 45, 13, 18, 44, 39, 26,
         27])
    X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)  # 把两个数据整合成一个n行2列的数组
    # 应用算法，训练
    k_means = K_Means(k=4)
    k_means.fit(X)
    # 绘图
    plt.plot()
    for i, feature in enumerate(X):
        classification = k_means.predict(feature)
        plt.plot(feature[0], feature[1], color=colors[classification], marker=markers[classification], ls='None')
    plt.xlim([-10, 60])
    plt.ylim([-10, 60])
    plt.show()
