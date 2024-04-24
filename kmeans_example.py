#clustering dataset
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

x1=np.array([3,1,1,2,1,6,6,6,5,6,7,8,9,8,9,9,8])
x2=np.array([5,4,6,6,5,8,6,7,6,7,1,2,1,2,3,2,3])

#viusalize the data
plt.plot()
plt.xlim([0,10])
plt.ylim([0,10])
plt.title('Dataset')
plt.scatter(x1,x2)
plt.show()

X=np.array(list(zip(x1,x2))).reshape(len(x1),2)#把两个数据整合成一个n行2列的数组？？
print(X)
colors=['b','g','r']
markers=['o','v','s']

#KMeans algorithm
K=3
kmeans_model=KMeans(n_clusters=K).fit(X)
print(kmeans_model.labels_)

plt.plot()
for i,l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i],x2[i],color=colors[l],marker=markers[l],ls='None')
    plt.xlim([0,10])
    plt.ylim([0,10])
plt.show()

#create new plot and data
plt.plot()
X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
colors=['b','g','r']
markers=['o','v','s']

#k means determine k
distortions=[]
K=range(1,10)
for k in K:
    kmeanModel= KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X,kmeanModel.cluster_centers_,'euclidean'),axis=1))/X.shape[0])

#Plot the elbow
plt.plot(K,distortions,'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
