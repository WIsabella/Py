import numpy as np
import operator


def DataSet():
     group=np.array([[3,21],[2,36],[8,33],[41,8],[29,4],[39,16]])
     labels=['娱乐日','娱乐日','娱乐日','锻炼日','锻炼日','锻炼日']
     return group,labels
def KNN(in_x,x_labels,y_labels,k):
     x_labels_size = x_labels.shape[0]
     distances = (np.tile(in_x,(x_labels_size,1))-x_labels)**2
     ad_distances = distances.sum(axis=1)
     sq_distances = ad_distances**0.5
     ed_distances = sq_distances.argsort()
     classdict={}
     for i in range(k):
         voteI_label = y_labels[ed_distances[i]]
         classdict[voteI_label] = classdict.get(voteI_label,0)+1
     sort_classdict = sorted(classdict.items(),key=operator.itemgetter(1),reverse=True)
     return sort_classdict[0][0]
if __name__ == '__main__':
    group,labels=DataSet()
    test_x=[50,20]
    print("输入的数据所对应的类型是：{}".format(KNN(test_x,group,labels,3)))

