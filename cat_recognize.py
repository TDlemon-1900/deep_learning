import numpy as np  # 是用python进行科学计算的基本软件包
import matplotlib.pyplot as plt  # 用于在python中绘制图表
import h5py  # 是与h5文件中存储的数据进行交互的常用软件包
import pylab

from lr_utils import load_dataset

# 把数据集中的数据加载到主程序中
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 查看特定图像
# index = 21
# plt.imshow(train_set_x_orig[index])
# pylab.show()

# 查看训练集中的标签
# print("{} = {}".format("train_set_y", str(train_set_y)))

# 查看训练集里面加载的数据
# train_set_x_orig是一个维度为（m_train, num_px,num_px, 3）的数组
m_train = train_set_y.shape[1]  # 训练集中的图片数量
m_test = test_set_y.shape[1]  # 测试集中的图片数量
num_px = train_set_x_orig.shape[1]  # 训练、测试集里面的图片的宽度和高度

print("训练集的数量: m_train = {}".format(str(m_train)))
print("测试集的数量: m_test = {}".format(str(m_test)))
print("每张图片的宽/高: num_px = {}".format(str(num_px)))
print("每张图片的大小: {}, {}, 3".format(str(num_px), str(num_px)))
print("训练集图片的维数: {}".format(str(train_set_x_orig.shape)))
print("训练集标签的维数: {}".format(str(train_set_y.shape)))
print("测试集图片的维数: {}".format(str(test_set_x_orig.shape)))
print("测试集标签的维数: {}".format(str(test_set_y.shape)))

"""
为了方便我们要把维度为（64，64，3）的numpy数组重新构造为（64x64x3，1）的数组，要乘以3的原因是 \
每张图片是由64x64像素构成的，而每个像素点由（R，G，B）三原色组成，所以要乘3。在此之后，训练集和 \
测试集是一个numpy数组，【每列代表一个平坦的图像】，应该有m_train和m_test列。
"""

"""
将数组变为209行的矩阵（因为训练集里有209张图片），用-1表示列代表让程序自己运算，最后得出是64*64*3 = 12288列, \
最后再使用转置，这样就变成了12288行，209列。
x = [x(1) x(2) ... x(m)]
"""
# 训练集的维度降低并转置
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
# 测试集的维度降低并转置
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("训练集降维最后的维度： {}".format(str(train_set_x_flatten.shape)))
print("训练集标签的维数: {}".format(str(train_set_y.shape)))
print("测试集降维之后的维度: {}".format(str(test_set_x_flatten.shape)))
print("测试集标签的维数 : {}".format(str(test_set_y.shape)))

"""
为了表示彩色图像，必须为每个像素指定红色，绿色和蓝色同通道，因此像素值实际上是从0到 \
255范围内的三个数字的向量。
机器学习中一个常见的预处理步骤是对数据进行居中和标准化，也就是减去每个示例中整个numpy \
数组的平均值，然后将每个示例除以整个numpy的标准差。但对于图片数据集，可以直接/255 \
让标准化的数据位于[0,1]之间
"""

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

"""
构建神经网络的步骤：
1.定义模型结构 （例如输入的特征向量）
2.初始化模型参数
3.循环
    3.1 计算当前损失（正向传播）
    3.2 计算当前梯度（反向传播）
    3.3 更新参数 （梯度下降）
"""


# 定义sigmoid函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))  # 1 / 1 + e^-z
    return s


# 初始化参数
def init_with_zeros(dim):
    """
    此函数构建一个维度为（dim,1)的向量，并将b初始化为0
    :param dim: w矢量的大小
    :return: 维度为（dim,1)的初始化向量
            b初始化的标量（对应bias）
    """
    print("dim = ", dim)
    w = np.zeros(shape=(dim, 1))
    b = 0

    # 确保w的向量维度是否为dim
    assert (w.shape == (dim, 1))
    # 确保bias的s数据类型是正确的
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# 进行前向传播和反向传播学习参数
def propagate(w, b, X, Y):
    """
    实现前向传播（forward propagation)和反向传播（back propagation)并更新参数
    :param w: 权重 （num_px * num_px * 3, 1)
    :param b: bias
    :param X: 训练样本 （num_px * pum * 3, 训练数量）
    :param Y: 真实的标签矢量 （0，1）
    :return:
        cost: 逻辑回归的负对数似然成本
        dw: w的相对损失梯度
        db：相对b的损失
    """

    m = X.shape[1]

    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b)  # (1, n)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # 计算成本

    # 反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)  # (dim, 1)
    db = (1 / m) * np.sum(A - Y)
    print((A- Y).shape)
    print(dw.shape)

    # 判断各数据数据是否正确
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # 创建一个字典，把dw和db保存起来
    grads = {
        "dw": dw,
        "db": db
    }

    return (grads, cost)


# 通过最小化成本函数J来学习w和b。对于参数w1, 更新规则是： w1 = w1 - lr*dw1
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    此函数通过梯度下降算法优化w和b

    :param w: 权重 （num_px * num_px * 3, 1)
    :param b: bias
    :param X: 维度为（num_px * num_px * 3, 1)
    :param Y: 真正的标签矢量，（如果是猫则为1，否则为0，矩阵维度为1
    :param num_iterations: 优化循环的迭代次数
    :param learning_rate: 学习率
    :param print_cost: 每100步打印一次损失值
    :return:
        params: 计算当前参数的成本和梯度，使用propagate
        使用w和b更新参数

    1) 计算当前参数的成本和梯度，使用propogate（）
    2）使用w和b的梯度下降法更新参数
    """

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)
        # 打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数 = {}, 误差值 = {}".format(i, cost))

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return (params, grads, costs)


# optimize函数会输出已学习的w和b的值，可以使用w和b来预测数据集X的标签

"""
预测函数predict：
    1）计算y^ = A = simoid(w^TX + b)
    2) 将a的值变为0或者为1
"""


def predict(w, b, X):
    """
    使用学习逻辑回归参数logistic (w, b)预测标签是0还是1

    :param w: 权重
    :param b: bias
    :param X: [x(1) x(2) ... x(m)]
    :return:
        Y_prediction: 包含X中所有图片的预测[0|1]
    """

    m = X.shape[1]  # 图片的数量
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计算猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# 将所有函数整个到model中
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, \
          learning_rate=0.5, print_cost=False):
    """
    调用之前的函数实现逻辑回归模型

    :param X_train: 大小为(num_px * num_px * 3, m_train)的训练集
    :param Y_train: 大小为(1， m_train)的训练集标签
    :param X_test: 大小为(num_px * num_px * 3, m_train)的测试集
    :param Y_test: 大小为(1， m_train)的测试集标签
    :param num_iterations: 优化参数的次数
    :param learning_rate: 学习率
    :param print_cost: 设置为true以每100次迭代打印成本
    :return:
        d: 包含有关模型的信息的字典
    """

    w, b = init_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, \
                                        print_cost)

    # 从字典中检索参数w和b
    w, b = parameters["w"], parameters["b"]

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("训练集准确性 : {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("测试集准确性 : {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return d


def paint(d):
    costs = np.squeeze[d["costs"]]
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate = {}".format(str(d["learning_rate"])))
    plt.show()


# 不同的learning_rate下曲线的对比
def main():
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is {}".format(str(i)))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, \
                               num_iterations=1500, learning_rate=i, print_cost=False)

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations')

    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    # pylab.show()


if __name__ == "__main__":
    main()
