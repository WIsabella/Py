from __future__ import print_function
import numpy as np
from keras.datasets import mnist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
batch_size = 128
num_classes = 10
epochs = 1

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        self.last_input = X
        self.last_activation = np.dot(X, self.weights) + self.biases
        if self.activation is not None:
            return self.activation(self.last_activation)
        return self.last_activation


class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def forward(self, X):
        self.last_input = X
        self.mask = np.random.rand(*X.shape) < self.dropout_rate
        return X * self.mask / self.dropout_rate


class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.cache = None

    def update(self, weights, grads):
        if self.cache is None:
            self.cache = [np.zeros_like(weight) for weight in weights]
        for i in range(len(weights)):
            self.cache[i] = self.rho * self.cache[i] + (1 - self.rho) * grads[i] ** 2
            weights[i] -= self.learning_rate * grads[i] / (np.sqrt(self.cache[i]) + self.epsilon)
        return weights


class SimpleNN:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def compile(self, optimizer, loss, metrics=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def loss_function(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def fit(self, x_train, y_train, batch_size, epochs, verbose, validation_data):
        for epoch in range(epochs):
            for i in range(0, len(x_train), batch_size):
                X_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                # 前向传播
                y_pred = self.forward(X_batch)

                # 反向传播
                grad = (y_pred - y_batch) / len(y_batch)
                for j in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[j]
                    if isinstance(layer, DenseLayer):
                        if j != len(self.layers) - 1:
                            grad = np.dot(grad, self.layers[j + 1].weights.T)
                        grad *= (y_pred * (1 - y_pred))
                        grad_weights = np.dot(X_batch.T, grad) / len(X_batch)
                        layer.weights -= self.optimizer.learning_rate * grad_weights
                        grad = np.dot(grad, layer.weights.T)

                # 打印损失
                loss = self.loss_function(y_batch, y_pred)
                print(f'Epoch {epoch + 1}/{epochs}, Batch {i//batch_size + 1}/{len(x_train)//batch_size}, Loss: {loss:.4f}')

    def evaluate(self, x_test, y_test):
        y_pred_test = self.forward(x_test)
        test_loss = -np.sum(y_test * np.log(y_pred_test + 1e-10)) / len(y_test)
        return test_loss

    def predict(self, sample):
        y_pred = self.forward(sample)
        return np.argmax(y_pred, axis=1)


# 数据加载与预处理
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


# 编译模型
def compile_model(model, optimizer, loss):
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


# 训练模型
# 训练模型
# 训练模型
# 训练模型
def train_model(model, x_train, y_train, batch_size, epochs, verbose, validation_data):
    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            X_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # 前向传播
            y_pred = model.forward(X_batch)

            # 反向传播
            grad = (y_pred - y_batch) / len(y_batch)
            for j in range(len(model.layers) - 1, -1, -1):
                layer = model.layers[j]
                if isinstance(layer, DenseLayer):
                    if j != len(model.layers) - 1:
                        grad = np.dot(grad, model.layers[j + 1].weights.T)
                    grad *= (y_pred * (1 - y_pred))
                    grad_weights = np.dot(X_batch.T, grad) / len(X_batch)
                    layer.weights -= model.optimizer.learning_rate * grad_weights
                    grad = np.dot(grad, layer.weights.T)

            # 打印损失
            loss = model.loss_function(y_batch, y_pred)
            print(f'Epoch {epoch + 1}/{epochs}, Batch {i//batch_size + 1}/{len(x_train)//batch_size}, Loss: {loss:.4f}')





# 评估模型
def evaluate_model(model, x_test, y_test):
    y_pred_test = model.forward(x_test)
    test_loss = -np.sum(y_test * np.log(y_pred_test + 1e-10)) / len(y_test)
    return test_loss

# 预测样本
def predict_sample(model, sample):
    y_pred = model.forward(sample)
    return np.argmax(y_pred, axis=1)


# 独热编码转换函数
def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]


# softmax 激活函数
def softmax(X):
    exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)
def relu(X):
    return np.maximum(0, X)

# 加载数据
(x_train, y_train), (x_test, y_test) = load_data()

# 定义损失函数
loss = 'categorical_crossentropy'

# 建立模型
model = SimpleNN()
model.add(DenseLayer(784, 512, activation=relu))
model.add(DropoutLayer(0.2))
model.add(DenseLayer(512, 512, activation=relu))
model.add(DropoutLayer(0.2))
model.add(DenseLayer(512, 10, activation=softmax))

# 编译模型
optimizer = RMSpropOptimizer(learning_rate=0.001)
compile_model(model, optimizer, loss)

# 训练模型
train_model(model, x_train, y_train, batch_size=128, epochs=1, verbose=2, validation_data=(x_test, y_test))

# 评估模型
test_loss = evaluate_model(model, x_test, y_test)
print('Test loss:', test_loss)

# 预测样本
one_sample = x_test[-1].reshape(1, 784)
predicted_class = predict_sample(model, one_sample)
print('Predicted class:', predicted_class)
