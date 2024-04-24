from __future__ import print_function
import keras
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 激活函数：ReLU
def relu(x):
    return np.maximum(0, x)

# 激活函数：Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 提高数值稳定性
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 交叉熵损失函数
def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-10  # 避免log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # 限制在(epsilon, 1-epsilon)范围内
    return -np.sum(y_true * np.log(y_pred), axis=1)

# RMSprop优化器
class RMSprop:
    def __init__(self, lr=0.001, rho=0.9, epsilon=None, decay=0.0):
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay

    def __str__(self):
        return "RMSprop(lr={}, rho={}, epsilon={}, decay={})".format(
            self.lr, self.rho, self.epsilon, self.decay
        )

    def update_weights(self, weights, gradients, cache=None):
        if cache is None:
            cache = np.zeros_like(weights)
        rho = self.rho
        epsilon = self.epsilon
        lr = self.lr
        decay = self.decay
        for i in range(len(weights)):
            cache[i] = rho * cache[i] + (1 - rho) * gradients[i] ** 2
            weights[i] -= (lr / (np.sqrt(cache[i]) + epsilon)) * gradients[i]
        if decay:
            lr *= 1.0 / (1.0 + decay)
        return weights, cache

# 全连接层
class Dense:
    def __init__(self, units, activation=None, input_shape=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.weights = None
        self.bias = None

    def initialize_weights(self):
        input_dim = self.input_shape[1]
        self.weights = np.random.randn(input_dim, self.units) * np.sqrt(2. / input_dim)
        self.bias = np.zeros(self.units)

    def forward(self, x):
        if self.weights is None:
            self.initialize_weights()
        self.last_input = x
        return np.dot(x, self.weights) + self.bias

# Dropout层
class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, x):
        self.last_input = x
        return np.where(np.random.rand(*x.shape) < self.rate, 0, x)

# 自定义顺序模型
class CustomSequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def summary(self):
        print("Model Summary")
        for i, layer in enumerate(self.layers):
            print("Layer {}: {}".format(i + 1, layer.__class__.__name__))

    def compile(self, loss, optimizer, metrics):
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        print("Model Compiled")

    def fit(self, x, y, batch_size, epochs, verbose, validation_data):
        print("Fitting the model...")
        history = {"loss": [0.1, 0.08, 0.06], "accuracy": [0.7, 0.8, 0.9]}  # 示例的训练历史记录字典
        return history

    def evaluate(self, x, y, verbose):
        print("Evaluating the model...")

    def predict(self, one_sample):  # 修改 predict 函数参数
        predicted = one_sample  # 这里暂时仅为示例，应该使用类内部的层进行前向传播
        predicted_class = np.argmax(predicted, axis=1)
        return predicted, predicted_class

# 加载数据
def load_data(num_classes):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test), num_classes  # 返回 num_classes

# 构建模型
def build_model():
    model = CustomSequential()
    model.add(Dense(512, activation=relu, input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation=relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=softmax))
    model.summary()
    return model

# 编译模型
def compile_model(model, loss, optimizer, metrics):
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

# 训练模型
def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(x_test, y_test))
    return history

# 评估模型
# 显示样本
def display_sample(one_sample):
    image_sample = one_sample.reshape(28, 28)
    plt.figure(1, figsize=(3, 3))
    plt.imshow(image_sample, cmap="gray_r", interpolation='nearest')
    plt.show()
# 评估模型
def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    return score

# 预测样本
def predict_sample(model, one_sample):
    predicted = model.predict(one_sample)
    predicted_class = np.argmax(predicted)  # 获取最高概率的索引
    return predicted, predicted_class

if __name__ == "__main__":
    batch_size = 128
    num_classes = 10
    epochs = 1

    # 加载数据
    (x_train, y_train), (x_test, y_test) = load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # 构建模型
    model = build_model()

    # 编译模型
    compile_model(model)

    # 训练模型
    history = train_model(model, x_train, y_train, x_test, y_test)

    # 评估模型
    eval_score = evaluate_model(model, x_test, y_test)
    if eval_score is not None:
        print('Test loss:', eval_score[0])
        print('Test accuracy:', eval_score[1])
    else:
        print("Evaluation failed.")

    # 预测样本
    one_sample = x_test[-1].reshape(1, 784)
    pred_prob, pred_class = predict_sample(model, one_sample)
    print('Predicted probabilities:', pred_prob)
    print('Predicted class:', pred_class)

    # 显示样本
    display_sample(one_sample)
