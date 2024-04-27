
from __future__ import print_function
import keras
from tensorflow.keras.datasets import mnist
from keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
#设置一些参数
batch_size = 128
num_classes = 10
epochs = 1

#装载数据，切分成训练数据集、测试数据集
# the data, shuffled and split between train and test sets
# keras中的mnist数据集已经被划分成了60,000个训练集，10,000个测试集的形式，按
#以下格式调用即可
(x_train, y_train), (x_test, y_test) = mnist.load_data()##[60000,28,28]  x数据，y标签

#转换成784列(784个特征)，将原本的60000*28*28的三维向量，转换为60000*784的二#维向量
x_train = x_train.reshape(60000, 784)# 重新定义数据格式
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')# 数据类型转换
x_test = x_test.astype('float32')
#把像素灰度转换成[0,1]之间的浮点数
x_train /= 255
x_test /= 255
#print(x_train.shape[0], 'train samples')# 60000 train samples
#print(x_test.shape[0], 'test samples')# 10000 test samples

# convert class vectors to binary class matrices
#print (y_train )
#print (y_test )

# #转one-hot标签
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#print (y_train )
#print (y_test )

#建立机器学习模型（添加层）
model = Sequential()#建立顺序模型,即前向反馈神经网络
'''
模型需要知道输入数据的shape，
因此，Sequential的第一层需要接收一个关于输入数据shape的参数，
后面的各个层则可以自动推导出中间数据的shape，
因此不需要为每个层都指定这个参数
'''
model.add(Dense(512, activation='relu', input_shape=(784,)))#普通神经网络层,512个神经元
model.add(Dropout(0.2))#利用Dropout技术，避免过拟合

model.add(Dense(512, activation='relu'))#普通神经网络层,512个神经元
model.add(Dropout(0.2))#利用Dropout技术，避免过拟合

model.add(Dense(10, activation='softmax'))#输出层，10个分类
# 输出模型的整体信息
# 总共参数数量为784*512+512 + 512*512+512 + 512*10+10 = 669706
model.summary()

#编译
'''
配置模型的学习过程
compile接收三个参数：
1. 损失函数loss：参数为模型试图最小化的目标函数，可为预定义的损失函数，
如categorical_crossentropy、mse，也可以为一个损失函数
2. 优化器optimizer：参数可指定为已预定义的优化器名，如rmsprop、adagrad，
或一个Optimizer类对象，如此处的RMSprop()
3. 评价指标：对于分类问题，一般将该列表设置为metrics=['accuracy']（准确率，精度，召回率）
'''
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

#训练
'''
训练模型
batch_size：指定梯度下降时每个batch包含的样本数
epoch：训练的轮数
verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为epoch输出一行记录
validation_data：指定验证集
fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，
如果有验证集的话，也包含了验证集的这些指标变化情况
'''
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(x_test, y_test))

#评估
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#预测
one_sample = x_test[-1]
print (one_sample.shape)
one_sample = one_sample.reshape(1,784)
print (one_sample.shape)

#显示一个样本
image_sample = one_sample.reshape(28,28)

print (image_sample.shape)
import matplotlib.pyplot as plt
plt.figure(1, figsize=(3, 3))
plt.imshow(image_sample, cmap="gray_r", interpolation='nearest')
plt.show()

#显示预测结果
predicted = model.predict(one_sample)
predicted_class = np.argmax(predicted, axis=1)  # 获取最高概率的索引
print(predicted)
print(predicted_class)