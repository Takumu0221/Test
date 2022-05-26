import math

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DATA_NUM = 10000
DIV = 5
EPOCH = 10
LR = 0.00001
HALF_NODES = 100


def sigmoid(x):
    beta = 1.0
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-beta * x))
    else:
        return np.exp(beta * x) / (np.exp(beta * x) + 1.0)


def derivative_sigmoid(x):
    return x * (1.0 - x)


def my_func(x1, x2):
    # return ((x1 ** 2 * x2 ** 2) ** (1 / 3)) * math.cos(x1 * x2)
    return x1 + x2


def calc_loss(z: np.ndarray, t: np.ndarray):
    loss = np.power(z - t, 2) / 2
    return np.sum(loss)


def MSE(z: np.ndarray, t: np.ndarray):
    loss = np.power(z - t, 2)
    return loss.mean()


def MAE(z: np.ndarray, t: np.ndarray):
    loss = np.abs(z - t)
    return loss.mean()


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
        self.act_func = sigmoid
        self.der_act_func = derivative_sigmoid

        # 重みの初期化
        self.w_kj = np.random.rand(self.output_nodes, self.half_nodes)
        self.w_ji = np.random.rand(self.half_nodes, self.input_nodes)

    def feed_forward(self, x):
        """シグマをそのまま記述"""
        y = np.zeros(self.half_nodes)
        for j in range(self.half_nodes):
            for i in range(self.input_nodes):
                y[j] += x[i] * self.w_ji[j, i]
            y[j] = sigmoid(y[j])

        z = np.zeros(self.output_nodes)
        for k in range(self.output_nodes):
            for j in range(self.half_nodes):
                z[k] += y[j] * self.w_kj[k, j]

        return z

    def back_prop(self, x, t):
        """シグマをそのまま記述"""
        y = np.zeros(self.half_nodes)
        for j in range(self.half_nodes):
            for i in range(self.input_nodes):
                y[j] += x[i] * self.w_ji[j, i]
            y[j] = sigmoid(y[j])
            # if np.isnan(y[j]):
            #     pass

        z = np.zeros(self.output_nodes)
        for k in range(self.output_nodes):
            for j in range(self.half_nodes):
                z[k] += y[j] * self.w_kj[k, j]

        loss = z - t

        # 重みの更新
        for k in range(self.output_nodes):
            for j in range(self.half_nodes):
                self.w_kj[k, j] += self.lr * loss[k] * derivative_sigmoid(z[k]) * y[j]
                # if np.isnan(self.w_kj[k, j]) or np.isinf(self.w_kj[k, j]):
                #     pass
        for j in range(self.half_nodes):
            for i in range(self.input_nodes):
                forward = 0
                for k in range(self.output_nodes):
                    forward += self.w_kj[k, j] * loss * derivative_sigmoid(z[k])
                self.w_ji[j, i] += self.lr * forward * derivative_sigmoid(y[j]) * x[i]
                # if np.isnan(self.w_ji[j, i]) or np.isinf(self.w_ji[j, i]):
                #     pass

        return z


def cross_validation(
        input_data,
        correct_data,
        div: int):
    loss_mse_list = []
    loss_mae_list = []
    for i in range(div):
        # ニューラルネットワーク
        nn = NeuralNetwork(
            half_nodes=HALF_NODES,
            lr=LR,
        )

        # 訓練データとテストデータの振り分け
        idx = list(range(i * int(DATA_NUM / div), (i + 1) * int(DATA_NUM / div)))
        idx_b = np.ones(DATA_NUM, dtype=bool)
        idx_b[idx] = False
        training_idata = input_data[idx_b, :]
        training_cdata = correct_data[idx_b, :]

        test_idata = input_data[idx[0]:idx[-1] + 1, :]
        test_cdata = correct_data[idx[0]:idx[-1] + 1, :]

        # 学習
        for e in range(EPOCH):
            loss = 0
            data_size = len(training_idata)
            for j in range(data_size):
                z = nn.back_prop(training_idata[j], training_cdata[j])
                loss += MSE(z, training_cdata[j])
            print(f'epoch{e} loss:{loss / data_size}')

        # テスト
        loss_mse = 0
        loss_mae = 0
        data_size = len(test_idata)
        output_data = []
        for j in range(data_size):
            z = nn.feed_forward(test_idata[j])
            loss_mse += MSE(z, test_cdata[j])
            loss_mae += MAE(z, test_cdata[j])
            output_data.append(z)
        print(f'test mse loss:{loss_mse / data_size}, test mae loss:{loss_mae / data_size}')
        loss_mse_list.append(loss_mse / data_size)
        loss_mae_list.append(loss_mae / data_size)

        plot(test_idata, output_data, test_cdata)

    print(f'cross validation result mse:{sum(loss_mse_list) / div} mae:{sum(loss_mae_list) / div}')


def plot(input_data, output_data, correct_data):
    """学習済みのモデルと目標関数をプロット"""
    # Figureを追加
    fig = plt.figure(figsize=(8, 8))

    # 3DAxesを追加
    ax = fig.add_subplot(111, projection='3d')

    # Axesのタイトルを設定
    ax.set_title("learn results", size=20)

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


def main():
    # データの準備
    N = 1
    input_data = np.concatenate([np.random.rand(DATA_NUM, 2) * 2 * N - 1 * N, np.ones((DATA_NUM, 1), dtype=float)], axis=1)
    correct_data = np.array([[my_func(_[0], _[1])] for _ in input_data])

    # 交差検証
    cross_validation(input_data, correct_data, DIV)


if __name__ == '__main__':
    main()
