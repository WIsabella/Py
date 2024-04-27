from __future__ import print_function
#导入keras库
import keras
from keras import mnist
from keras import Sequential
from keras import Dense, Dropout, Flatten
from keras import Conv2D, MaxPooling2D
from keras.api import backend as K

#设置一些参数
batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

#装载数据，切分成训练数据集、测试数据集
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#把像素灰度转换成[0,1]之间的浮点数
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 建立机器学习模型
model = Sequential()  # 建立顺序模型,即前向反馈神经网络
# 第一个卷积层
# input_shape 输入平面
# filters 卷积核/滤波器个数32
# kernel_size 卷积窗口大小
# activation 激活函数
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))  # 2维卷积层

model.add(Conv2D(64, (3, 3), activation='relu'))  # 2维卷积层

model.add(MaxPooling2D(pool_size=(2, 2)))  # 子采样层
model.add(Dropout(0.25))  # 利用Dropout技术，避免过拟合

model.add(Flatten())  # 把输入数据压扁，即把多维向量变成一维向量

model.add(Dense(128, activation='relu'))  # 普通神经网络层，128个神经元
model.add(Dropout(0.5))  # 利用Dropout技术，避免过拟合

model.add(Dense(num_classes, activation='softmax'))  # 输出层，10个分类

#编译
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#训练
# verbose是屏显模式：就是说0是不屏显，1是显示一个进度条，2是每个epoch都显示#一行数据
# validation_data就是验证测试集
model.fit(x_train, y_train,
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
one_sample = one_sample.reshape(1,28,28,1)
print (one_sample.shape)

#显示一个样本
image_sample = one_sample.reshape(28,28)
print (image_sample.shape)
import matplotlib.pyplot as plt
plt.figure(1, figsize=(3, 3))
plt.imshow(image_sample, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

#显示预测结果
predicted = model.predict(one_sample)
predicted_class = model.predict_classes(one_sample, verbose=0)
print (predicted)
print (predicted_class)