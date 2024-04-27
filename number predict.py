import numpy as np
import matplotlib.pyplot as plt
# 加载 MNIST 数据集
def load_data():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return (x_train, y_train), (x_test, y_test)
# 数据预处理
def preprocess_data(x_train, x_test):
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0
    return x_train, x_test
# 将类别向量转换为二进制类矩阵（独热编码）
def to_categorical(y, num_classes):
    m = len(y)
    y_one_hot = np.zeros((m, num_classes))
    for i in range(m):
        y_one_hot[i, y[i]] = 1
    return y_one_hot
# 激活函数
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
# 前向传播
def forward_propagation(X, parameters):
    # 获取参数
    W1, b1, W2, b2, W3, b3 = parameters

    # 第一隐藏层
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    # 第二隐藏层
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)

    # 输出层
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3
# 计算损失
def compute_loss(A3, Y):
    m = Y.shape[0]
    loss = -1 / m * np.sum(Y * np.log(A3 + 1e-8))
    return loss
# 反向传播
def backward_propagation(X, Y, parameters, caches):
    m = X.shape[0]
    W1, b1, W2, b2, W3, b3 = parameters
    Z1, A1, Z2, A2, Z3, A3 = caches

    dZ3 = (A3 - Y) / m
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    gradients = (dW1, db1, dW2, db2, dW3, db3)

    return gradients
# 更新参数
def update_parameters(parameters, gradients, learning_rate):
    W1, b1, W2, b2, W3, b3 = parameters
    dW1, db1, dW2, db2, dW3, db3 = gradients

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

    parameters = (W1, b1, W2, b2, W3, b3)

    return parameters
# 训练模型
def train_model(x_train, y_train, learning_rate=0.001, num_epochs=10, batch_size=128, hidden_layer_size=512):
    # 获取数据形状和类别数
    m, n = x_train.shape
    num_classes = y_train.shape[1]

    # 初始化参数
    np.random.seed(1)
    W1 = np.random.randn(n, hidden_layer_size) * np.sqrt(2.0 / n)
    b1 = np.zeros((1, hidden_layer_size))
    W2 = np.random.randn(hidden_layer_size, hidden_layer_size) * np.sqrt(2.0 / hidden_layer_size)
    b2 = np.zeros((1, hidden_layer_size))
    W3 = np.random.randn(hidden_layer_size, num_classes) * np.sqrt(2.0 / hidden_layer_size)
    b3 = np.zeros((1, num_classes))

    parameters = (W1, b1, W2, b2, W3, b3)

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0  # 初始化本轮的总损失

        # 随机打乱训练数据
        permutation = np.random.permutation(m)
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i in range(0, m, batch_size):
            # 获取小批量数据
            x_batch = x_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            # 前向传播
            caches = forward_propagation(x_batch, parameters)
            Z1, A1, Z2, A2, Z3, A3 = caches

            # 计算损失
            loss = compute_loss(A3, y_batch)

            # 累加本轮的总损失
            total_loss += loss

            # 反向传播
            gradients = backward_propagation(x_batch, y_batch, parameters, caches)

            # 更新参数
            parameters = update_parameters(parameters, gradients, learning_rate)

        # 计算本轮的平均损失
        average_loss = total_loss / (m // batch_size)
        print("第{}轮: 损失 = {}".format(epoch + 1, average_loss))

    return parameters
# 预测
def predict(x_test, parameters):
    caches = forward_propagation(x_test, parameters)
    _, _, _, _, _, A3 = caches
    predictions = np.argmax(A3, axis=1)
    return predictions
# 加载数据
(x_train, y_train), (x_test, y_test) = load_data()

# 数据预处理
x_train, x_test = preprocess_data(x_train, x_test)

# 将标签转换为独热编码
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 训练模型
parameters = train_model(x_train, y_train, num_epochs=20)

# 在测试集上进行预测
predictions = predict(x_test, parameters)

# 计算准确率
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print("测试准确率:", accuracy)

# 在测试集上预测所有样本并显示一个随机样本
random_index = np.random.randint(len(x_test))
sample_image = x_test[random_index]
sample_label = np.argmax(y_test[random_index])
sample_prediction = np.argmax(forward_propagation(sample_image, parameters)[-1])
print("样本标签:", sample_label)
print("样本预测:", sample_prediction)

# 显示样本图像
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.title(f"样本标签: {sample_label}, 预测: {sample_prediction}")
plt.show()
