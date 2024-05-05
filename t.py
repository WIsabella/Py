import numpy as np
import matplotlib.pyplot as plt


# 加载 MNIST 数据集
def load_data():
    from tensorflow.keras.datasets import mnist  # 导入mnist数据集（从tensorflow中导入与我的环境配置有关）
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # 利用两个元组保存训练集train和测试集test
    return (x_train, y_train), (x_test, y_test)


# 数据预处理
def preprocess_data(x_train, x_test):
    x_train = x_train.reshape(-1, 784).astype('float32') / 255.0  #将图像修改为一维数组，-1表示根据维度自行判断，28*28=784
    x_test = x_test.reshape(-1, 784).astype('float32') / 255.0  #将灰度转化成01之间的浮点数
    return x_train, x_test


# 将类别向量转换为二进制类矩阵（独热编码）
def to_categorical(y, num_classes):  # y标签列表
    m = len(y)  #取长度
    y_one_hot = np.zeros((m, num_classes))  #创建一个形状为（样本数量，类型数）的全零数组，存储独热编码结果
    for i in range(m):  # 遍历
        y_one_hot[i, y[i]] = 1  #第i个样本相应的i位置设为1，其他保持为0
    return y_one_hot  # 返回这个数组


# 激活函数
def relu(x):  #经典relu激活函数，小于0赋值为0截断，大于0不变。
    return np.maximum(0, x)


def softmax(x):  # 将向量转换为概率分布的激活函数，确保输出概率综合为1（相当于概率）
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # np.max 输入向量x沿着样本方向取最大值，keepdims=True保持维度不变
    # exp_x计算输入向量x减去每行最大之后的e指数值，确保计算稳定，避免指数运算的溢出
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # 对每行指数值求和，得到每一行的归一化因子
    # 每个元素除以对应行的归一化因子，得到最终的Softmax输出


# Dropout函数：
def dropout(A, keep_prob):
    # 生成一个与A形状相同的布尔矩阵，元素值True 或者 False表示保留对应位置激活值
    D = np.random.rand(*A.shape) < keep_prob
    # 将A中与D对应False位置的元素置为0
    A *= D
    # 将剩余的激活值除以keep_prob保持激活值的期望不变
    A /= keep_prob
    #返回处理后的激活值矩阵A和对应dropout掩码矩阵D，在反向传播时可以恢复对应的梯度。
    return A, D


# 前向传播
def forward_propagation(X, parameters, keep_prob):  # 添加keep_prob参数用于控制dropout的保留概率
    # 获取参数
    W1, b1, W2, b2, W3, b3 = parameters  #从参数列表中解包得到神经网络的权重矩阵和偏置向量，每一层都要用

    # 第一隐藏层 先W1X+b1再relu一下，得到第一隐藏层的激活值
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    A1, D1 = dropout(A1, keep_prob)  # 应用dropout到第一隐藏层的激活值

    # 第二隐藏层 同理于第一
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    A2, D2 = dropout(A2, keep_prob)  # 应用dropout到第二隐藏层的激活值

    # 输出层
    Z3 = np.dot(A2, W3) + b3
    A3 = softmax(Z3)  # 使用softmax来输出：概率分布

    caches = (Z1, A1, D1, Z2, A2, D2, Z3, A3)  # 保存中间值以备反向传播

    return caches  # 输出每一层加权输入和激活值，后续继续使用


# 反向传播
def backward_propagation(X, Y, parameters, caches, keep_prob):  # 添加keep_prob参数
    m = X.shape[0]  # 先获取样本数量
    W1, b1, W2, b2, W3, b3 = parameters  # 解包权重矩阵和偏执向量
    Z1, A1, D1, Z2, A2, D2, Z3, A3 = caches  # 解包前向传播中的计算中间值

    dZ3 = (A3 - Y) / m  # 计算输出层激活值A3相对于损失函数的梯度
    dW3 = np.dot(A2.T, dZ3)  # W3的梯度
    db3 = np.sum(dZ3, axis=0, keepdims=True)  # b3的梯度

    dA2 = np.dot(dZ3, W3.T)  # A3梯度
    dA2 *= D2  # dropout的反向传播，只保留之前随机选择的节点
    dA2 /= keep_prob  # 缩放激活值，以保持期望值不变
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))  # 加权输入的梯度
    dW2 = np.dot(A1.T, dZ2)  # ...
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # ...

    dA1 = np.dot(dZ2, W2.T)
    dA1 *= D1
    dA1 /= keep_prob
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    gradients = (dW1, db1, dW2, db2, dW3, db3)  #将所有参数的梯度打包成一个元组

    return gradients  # 返回更新神经网络的参数

def update_parameters(parameters, gradients, learning_rate):  # learning_rate 学习率，控制参数更新的不长
    W1, b1, W2, b2, W3, b3 = parameters  # 俩参数
    dW1, db1, dW2, db2, dW3, db3 = gradients  # 参数梯度

    W1 -= learning_rate * dW1  # 更新第一层参数
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2  # 第二层
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3  # 输出层
    b3 -= learning_rate * db3

    parameters = (W1, b1, W2, b2, W3, b3)  # 参数继续打包

    return parameters
def compute_loss(A3, Y):  # A3概率分布二维数组（样本数量，类别数量）  Y标签，与A3形状相同
    m = Y.shape[0]  # 获取样本数量：标签的行数
    # 每个样本的预测概率分布A3和实际标签Y的对应元素逐个相乘再求和，计算交叉熵损失 1e-8避免log数值不稳定
    #求和结果除以样本数量，乘以-1得到最终损失值（标准）
    loss = -1 / m * np.sum(Y * np.log(A3 + 1e-8))
    return loss
# 训练模型
def train_model(x_train, y_train, learning_rate=0.001, num_epochs=10, batch_size=128, hidden_layer_size=512, keep_prob=0.8):
    m, n = x_train.shape  # 样本数量和特征数量
    num_classes = y_train.shape[1]  # 类别数量

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
            caches = forward_propagation(x_batch, parameters, keep_prob)  # 添加keep_prob参数
            Z1, A1, D1, Z2, A2, D2, Z3, A3 = caches

            # 反向传播
            gradients = backward_propagation(x_batch, y_batch, parameters, caches, keep_prob)  # 添加keep_prob参数

            # 更新参数
            parameters = update_parameters(parameters, gradients, learning_rate)

            # 计算损失
            loss = compute_loss(A3, y_batch)

            # 累加本轮的总损失
            total_loss += loss

        # 计算本轮的平均损失
        average_loss = total_loss / (m // batch_size)
        print("Epoch {}: Loss = {}".format(epoch + 1, average_loss))

    return parameters


def predict(x_test, parameters):
    caches = forward_propagation(x_test, parameters, 1.0)  # Keep probability为1.0，表示不使用dropout
    _, _, _, _, _, _, _, A3 = caches  # 只取A3即可
    predictions = np.argmax(A3, axis=1)  # 对输出层的激活值按行取最大值，得到预测结果。
    return predictions  # 标签返还

# 加载数据
(x_train, y_train), (x_test, y_test) = load_data()

# 数据预处理
x_train, x_test = preprocess_data(x_train, x_test)

# 将标签转换为独热编码
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# 训练模型
parameters = train_model(x_train, y_train, num_epochs=20, keep_prob=0.8)  # 添加keep_prob参数

# 在测试集上进行预测
predictions = predict(x_test, parameters)

# 计算准确率
accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
print("Test Accuracy:", accuracy)

# 在测试集上预测所有样本并显示一个随机样本
random_index = np.random.randint(len(x_test))
sample_image = x_test[random_index]
sample_label = np.argmax(y_test[random_index])
sample_prediction = np.argmax(forward_propagation(sample_image, parameters, 1.0)[-1])
print("Sample label:", sample_label)
print("Sample prediction:", sample_prediction)

# 显示样本图像
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.axis('off')
plt.title(f"Sample label: {sample_label}, Prediction: {sample_prediction}")
plt.show()
