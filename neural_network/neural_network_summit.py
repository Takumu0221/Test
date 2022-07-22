import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List

DATA_NUM = 10000
DIV = 5
EPOCH = 100
LR = 0.01
HALF_NODES = 100


def sigmoid(x):
    beta = 1.0
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-beta * x)), np.exp(beta * x) / (np.exp(beta * x) + 1.0))


def derivative_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def identity(x):
    return x

def derivative_identity(x):
    return 1


def relu(x):
    return np.maximum(0, x)


def derivative_relu(x):
    return np.where(x > 0, 1, 0)


def tanh_exp(x):
    return x * np.tanh(np.exp(x))


def derivative_tanh_exp(x):
    return np.tanh(np.exp(x)) - x * np.exp(x) * (np.square(np.tanh(np.exp(x))) - 1)


def my_func(x1, x2):
    return ((x1 ** 2 * x2 ** 2) ** (1 / 3)) * math.cos(x1 * x2 * 5)
    # return x1 + x2


def calc_loss(z: np.ndarray, t: np.ndarray):
    loss = np.power(z - t, 2) / 2
    return np.sum(loss)


class NeuralNetwork:
    def __init__(self,
                 input_nodes=3,
                 half_nodes=5,
                 output_nodes=1,
                 lr=0.001,
                 ):
        self.input_nodes = input_nodes
        self.half_nodes = half_nodes
        self.output_nodes = output_nodes
        self.lr = lr

        # 活性化関数
        self.act_func = {"h": tanh_exp, "o": identity}
        self.der_act_func = {"h": derivative_tanh_exp, "o": derivative_identity}

        # 重みの初期化
        self.w_kj = np.random.normal(0.0, 1.0 / math.sqrt(half_nodes), (self.output_nodes, self.half_nodes))
        self.w_ji = np.random.normal(0.0, 1.0 / math.sqrt(input_nodes), (self.half_nodes, self.input_nodes))

    def feed_forward(self, x: np.ndarray):
        """行列で計算する"""
        xT = x.T

        y = np.dot(self.w_ji, xT)
        y = self.act_func["h"](y)

        z = np.dot(self.w_kj, y)

        return z

    def back_prop(self, x, t):
        xT = np.array(x, ndmin=2).T
        tT = np.array(t, ndmin=2).T

        u = np.dot(self.w_ji, xT)
        y = self.act_func["h"](u)

        z = np.dot(self.w_kj, y)

        loss = (tT - z)
        loss_h = np.dot(self.w_kj.T, loss)

        self.w_kj += self.lr * np.dot((loss * self.der_act_func["o"](z)), y.T)
        self.w_ji += self.lr * np.dot((loss_h * self.der_act_func["o"](z)) * self.der_act_func["h"](u), xT.T)

        return z


def train_and_test(
        training_idata,
        training_cdata,
        test_idata,
        test_cdata,
        returned_dict,
        process_index,
        half_nodes=HALF_NODES,
):
    """与えられた入力と正解データで学習及びテストを行う"""
    # nn = NeuralNetwork(half_nodes=half_nodes, lr=LR)
    nn = NeuralNetwork(half_nodes=half_nodes, lr=LR)

    # 学習
    loss_list = []
    for e in range(EPOCH):
        loss = 0
        data_size = len(training_idata)
        for j in range(data_size):
            z = nn.back_prop(training_idata[j], training_cdata[j])
            loss += calc_loss(z, training_cdata[j])
        print(f'epoch{e} loss:{loss}') if process_index == 0 else None
        loss_list.append(loss)

    # テスト
    loss = 0
    data_size = len(test_idata)
    output_data = []
    for j in range(data_size):
        z = nn.feed_forward(test_idata[j])
        loss += calc_loss(z, test_cdata[j])
        output_data.append(z)
    print(f'test loss:{loss}')

    plot(test_idata, output_data, test_cdata)
    plot_loss(loss_list)

    returned_dict[process_index] = {'sse': loss}


def plot(input_data, output_data, correct_data):
    """学習済みのモデルと目標関数をプロット"""
    # Figureを追加
    fig = plt.figure(figsize=(8, 8))

    # 3DAxesを追加
    ax = fig.add_subplot(111, projection='3d')

    # Axesのタイトルを設定
    # ax.set_title("learn results", size=20)

    # 軸ラベルを設定
    ax.set_xlabel("x1", size=14, color="r")
    ax.set_ylabel("x2", size=14, color="r")
    ax.set_zlabel("f(x1,x2)", size=14, color="r")

    # 軸目盛を設定
    # ticks = [-1.0, -0.5, 0.0, 0.5, 1.0] * 10
    # ax.set_xticks(ticks)
    # ax.set_yticks(ticks)

    for z, color in [[output_data, 'blue'], [correct_data, 'red']]:
        # 変数に代入
        x = input_data[:, 0]
        y = input_data[:, 1]
        z = z

        # 曲線を描画
        ax.scatter(x, y, z, s=40, c=color)

    plt.show()


def plot_loss(loss_list: List[float]):
    """誤差の推移をプロット"""
    num = len(loss_list)

    x = np.arange(1, num + 1)
    y = np.array(loss_list)

    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.plot(x, y)
    plt.show()
