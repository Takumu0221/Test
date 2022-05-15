import numpy as np
import math


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
        pass

    def back_prop(self):
        pass


def main():
    # データの準備
    DATA_NUM = 10000
    input_data = np.concatenate([np.random.rand(2, DATA_NUM) * 20 - 10, np.ones((2, DATA_NUM))])
    output_data = np.array([my_func(_[0], _[1]) for _ in input_data])

    # ニューラルネットワーク
    nn = NeuralNetwork(half_nodes=5)

    # 学習
    epoch = 5
    for e in range(epoch):
        nn.feed_forward(input_data)
        nn.back_prop()

    # 交差検証
    N = 5
    for i in range(5):
        pass


if __name__ == '__main__':
    main()
