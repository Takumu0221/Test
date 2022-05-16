import numpy as np
import math

DATA_NUM = 10000
DIV = 5
EPOCH = 10

def sigmoid(x):
    beta = 1
    return 1 / (1 + math.exp(-beta * x))


def derivative_sigmoid(x):
    return x * (1.0 - x)


def my_func(x1, x2):
    return ((x1 ** 2 * x2 ** 2) ** (1 / 3)) * math.cos(x1 * x2)


class NeuralNetwork:
    def __init__(self,
                 input_nodes=3,
                 half_nodes=5,
                 output_nodes=1,
                 lr=0.01,
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

        z = np.zeros(self.output_nodes)
        for k in range(self.output_nodes):
            for j in range(self.half_nodes):
                z[k] += y[j] * self.w_kj[k, j]

        loss = t - z

        # 重みの更新
        for k in range(self.output_nodes):
            for j in range(self.half_nodes):
                self.w_kj[k, j] += self.lr * loss[k] * derivative_sigmoid(z[k]) * y[j]
        for j in range(self.half_nodes):
            for i in range(self.input_nodes):
                forward = 0
                for k in range(self.output_nodes):
                    forward += self.w_kj[k, j] * loss[k] * derivative_sigmoid(z[k])
                self.w_ji += self.lr * forward * derivative_sigmoid(y[j]) * x[i]

        return loss


def cross_validation(
        input_data,
        correct_data,
        nn: NeuralNetwork,
        div: int):

    for i in range(div):
        # 訓練データとテストデータの振り分け
        idx = list(range(i * int(DATA_NUM / div), (i+1) * int(DATA_NUM / div)))
        idx_b = np.ones(DATA_NUM, dtype=bool)
        idx_b[idx] = False
        training_idata = input_data[:, idx_b]
        training_cdata = correct_data[:, idx_b]

        test_idata = input_data[:, idx[0]:idx[-1]]
        test_cdata = correct_data[:, idx[0]:idx[-1]]

        # 学習
        for e in range(EPOCH):
            loss = 0
            data_size = len(training_idata)
            for j in range(data_size):
                loss += nn.back_prop(training_idata[j], training_cdata[j])
            print(f'epoch{e} loss:{loss / data_size}')

        # テスト
        loss = 0
        data_size = len(test_idata)
        for j in range(data_size):
            z = nn.feed_forward(test_idata)
            loss += test_cdata[j] - z[0]
        print(f'test loss:{loss / data_size}')


def main():
    # データの準備
    input_data = np.concatenate([np.random.rand(2, DATA_NUM) * 20 - 10, np.ones((2, DATA_NUM))])
    correct_data = np.array([my_func(_[0], _[1]) for _ in input_data])

    # ニューラルネットワーク
    nn = NeuralNetwork(half_nodes=5)

    # 交差検証
    cross_validation(input_data, correct_data, nn, DIV)


if __name__ == '__main__':
    main()
