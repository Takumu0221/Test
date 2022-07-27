import datetime
import json
import math
import os
import random
from multiprocessing import Process

import numpy as np
from typing import List


def rastrigin(x: np.ndarray) -> float:
    return sum(x * x - 10 * np.cos(2 * math.pi * x) + 10)


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
                 N=10000,  # 1世代の個体数
                 a=0.2,  # ブレンド交叉の幅係数
                 Pm=0.05,  # 突然変異確率
                 np=10,  # 親個体群数
                 nc=500,  # 子個体群数
                 dim=50,  # 次元数
                 min_value=-5.12,  # 個体値の最小値
                 max_value=5.12,  # 個体値の最大値
                 record=True, # 記録するかどうか
                 ):
        # 各種パラメタ
        self.N = N
        self.a = a
        self.Pm = Pm
        self.np = np
        self.nc = nc

        # 関数の設定
        self.dim = dim
        self.min_value = min_value
        self.max_value = max_value

        # 集団
        self.individual_group = []
        self.parent_group = []
        self.children_group = []

        # 記録
        self.result = float('inf')
        self.repeat_times = 0
        self.record = record
        self.d_path = None

    def run(self):
        self.reset()
        self.record_param()

        i = 0
        while self.repeat_times < 1000:
            # print(self.result)
            self.evaluate(self.individual_group)
            self.random_select()
            self.crossover()
            self.mutation()
            self.evaluate(self.children_group)
            self.select()

            # 記録
            new_result = min([I.fitness for I in self.individual_group])
            if self.result - new_result < 0.001:
                self.repeat_times += 1
            else:
                self.repeat_times = 0
                self.result = new_result
            self.record_results(i)
            i += 1

    def reset(self):
        # 個体の生成・初期化
        self.individual_group = [Individual() for _ in range(self.N)]

        # パラメタの初期化
        self.repeat_times = 0

        # 記録用ディレクトリのパスを設定
        now = datetime.datetime.now()
        self.d_path = f'results/GA/{now.strftime("%Y%m%d-%H%M%S%f")}'

    def set_param(self, param: dict):
        self.N = param['N']
        self.a = param['a']
        self.Pm = param['Pm']

    def evaluate(self, group: List[Individual]):
        # 与えられたリスト内の個体の適応度を計算
        for i in group:
            i.evaluate()

    def random_select(self):
        self.parent_group = []
        for i in range(self.np):
            l = len(self.individual_group)
            self.parent_group.append(self.individual_group.pop(random.randint(0, l - 1)))

    def crossover(self):
        self.children_group = []
        for i in range(self.nc):
            pa, pb = np.random.choice(self.parent_group, 2)
            degree = np.abs(pa.x - pb.x)
            max_values = np.max([pa.x, pb.x], axis=0) + self.a * degree
            min_values = np.min([pa.x, pb.x], axis=0) - self.a * degree

            child = Individual()
            child.x = np.random.uniform(min_values, max_values, self.dim)
            self.children_group.append(child)

    def mutation(self):
        for I in self.children_group:
            if random.random() < self.Pm:
                I.x += np.random.normal(0.0, 1.0, self.dim)
                I.x = np.clip(I.x, self.min_value, self.max_value)

    def select(self):
        selected = []
        sorted_group = sorted(self.children_group, key=lambda x: x.fitness, reverse=False)

        # エリート選択
        elite = sorted_group.pop(0)
        selected.append(elite)

        # ランキング選択
        prob = ranking_probability(self.nc - 1)
        rank = np.random.choice(sorted_group, size=self.np - 1, p=prob, replace=False)
        selected += list(rank)

        self.individual_group += selected

    def record_param(self):
        if not self.record:
            return None

        os.makedirs(self.d_path, exist_ok=True)
        filename = f'{self.d_path}/param.json'
        params = {
            'N': self.N,
            'a': self.a,
            'Pm': self.Pm,
            'np': self.np,
            'nc': self.nc,
        }
        with open(filename, 'w') as f:
            json.dump(params, f)

    def record_results(self, i: int):
        if not self.record:
            return None

        filename = f'{self.d_path}/fitness.csv'
        if not os.path.isfile(filename):
            with open(filename, 'w') as f:
                f.write('i,fitness\n')
        else:
            with open(filename, 'a') as f:
                f.write(f'{i},{self.result}\n')


def main():
    process_list = []
    for N in [1e3, 1e4, 1e5]:
        for a in [0.1, 0.2, 0.3, 0.5]:
            for Pm in [1e-1, 1e-2, 1e-3, 0]:
                GA = real_code_GA()
                param = {
                    'N': int(N),
                    'a': a,
                    'Pm': Pm,
                }
                GA.set_param(param)
                process = Process(target=GA.run)
                process_list.append(process)
                process.start()
                # print(GA.result)

    for p in process_list:
        p.join()


if __name__ == '__main__':
    main()
