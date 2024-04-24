#clustering dataset
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

x1 = np.array(
     [39, 9, 11, 44, 14, 8, 25, 0, 12, 45, 30, 32, 45, 15, 15, 6, 6, 46, 39, 13, 47, 1, 9, 17, 38, 34, 25, 2, 37, 9])
x2 = np.array(
     [19, 41, 43, 35, 6, 36, 4, 13, 34, 38, 13, 25, 44, 14, 44, 10, 6, 45, 2, 19, 48, 29, 49, 45, 13, 18, 44, 39, 26,
      27])

# plt.plot()
# plt.xlim([-10,60])
# plt.ylim([-10,60])
# plt.title('Dataset')
# plt.scatter(x1, x2)
# plt.show()

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2) #把两个数据整合成一个n行2列的数组
#print(X)
colors = ['blue', 'green', 'red','purple']
markers = ['o', 'v', 's','*']

# KMeans algorithm
K = 4
kmeans_model = KMeans(n_clusters=K).fit(X)
#print(kmeans_model.labels_)

plt.plot()
for i, l in enumerate(kmeans_model.labels_):
     plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
     plt.xlim([-10, 60])
     plt.ylim([-10, 60])
plt.show()

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
#distortions = []
#K = range(1, 10)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k).fit(X)
#    kmeanModel.fit(X)
#    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')
#plt.show()
