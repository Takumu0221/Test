import math
import random

import numpy as np
from typing import List


def rastrigin(x: np.ndarray) -> float:
    return sum(x * x - 10 * np.cos(2 * math.pi * x + 10))

def ranking_probability(N: int):
    probabilities = np.array([(1 / (r + 1)) ** 1.5 for r in range(1, N + 1)])
    total = sum(probabilities)
    return list(probabilities / total)

class Individual:
    def __init__(self,
                 dim=50,
                 max_value: float = 5.12,
                 min_value: float = -5.12,
                 ):
        self.x = (max_value - min_value) * np.random.rand(dim) + min_value
        self.fitness = 0

    def evaluate(self):
        self.fitness = rastrigin(self.x)
        return self.fitness


class real_code_GA:
    def __init__(self,
                 N=10,  # 1世代の個体数
                 a=0.2,  # ブレンド交叉の幅係数
                 Pm=0.05,  # 突然変異確率
                 np=10,  # 親個体群数
                 nc=500  # 子個体群数
                 ):
        # 各種パラメタ
        self.N = N
        self.a = a
        self.Pm = Pm
        self.np = np
        self.nc = nc

        # 集団
        self.individual_group = []
        self.parent_group = []
        self.children_group = []

    def init(self):
        # 個体の生成・初期化
        self.individual_group = [Individual() for _ in range(self.N)]

    def evaluate(self, group: List[Individual]):
        # 与えられたリスト内の個体の適応度を計算
        for i in group:
            i.evaluate()

    def select(self):
        selected = []
        sorted_group = sorted(self.children_group, key=lambda x: x.fitness, reverse=True)

        # エリート選択
        selected.append(sorted_group.pop(0))

        # ランキング選択
        prob = ranking_probability(self.nc - 1)
        rank = np.random.choice(sorted_group, size=self.np - 1, p=prob, replace=False)
        selected += list(rank)

        self.individual_group += selected

    def random_select(self):
        self.parent_group = []
        for i in range(self.np):
            l = len(self.individual_group)
            self.parent_group.append(self.individual_group.pop(random.randint(0, l-1)))

    def crossover(self):
        self.children_group = self.parent_group * int(self.nc / self.np)


def main():
    GA = real_code_GA()

    GA.init()

    for i in range(10):
        print(GA.individual_group[0].fitness)
        GA.evaluate(GA.individual_group)
        GA.random_select()
        GA.crossover()
        GA.select()


if __name__ == '__main__':
    main()
