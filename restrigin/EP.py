from copy import deepcopy

import numpy as np
from multiprocessing import Process

from GA import *

def sigmoid(x):
    beta = 1.0
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-beta * x)), np.exp(beta * x) / (np.exp(beta * x) + 1.0))

class Individual:
    def __init__(self,
                 dim=50,
                 max_value: float = 5.12,
                 min_value: float = -5.12,
                 ):
        self.x = (max_value - min_value) * np.random.rand(dim) + min_value
        self.fitness = 0
        self.value = 0

    def evaluate(self):
        self.fitness = rastrigin(self.x)
        return self.fitness


class EP:
    def __init__(self,
                 N=10000,  # 1世代の個体数
                 M=100,  # 生存価値を求める際に選択する個体数
                 dim=50,  # 次元数
                 min_value=-5.12,  # 個体値の最小値
                 max_value=5.12,  # 個体値の最大値
                 record=True,  # 記録するかどうか
                 ):
        # 各種パラメタ
        self.N = N
        self.M = M

        # 関数の設定
        self.dim = dim
        self.min_value = min_value
        self.max_value = max_value

        # 集団
        self.individual_group = []
        self.survivor_group = []

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
            self.duplicate()
            self.evaluate(self.survivor_group)
            self.evaluate_value()
            self.select()
            self.evaluate(self.individual_group)

            # 記録
            new_result = min([I.fitness for I in self.individual_group])
            if self.result - new_result < 0.01:
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
        self.d_path = f'results/EP/{now.strftime("%Y%m%d-%H%M%S%f")}'

    def set_param(self, param: dict):
        self.N = param['N']
        self.M = param['M']

    def evaluate(self, group: List[Individual]):
        # 与えられたリスト内の個体の適応度を計算
        for i in group:
            i.evaluate()

    def duplicate(self):
        duplicated = self.mutation([deepcopy(_) for _ in self.individual_group])
        self.survivor_group = self.individual_group + duplicated

    def mutation(self, group: List[Individual]):
        for I in group:
            I.x += sigmoid(I.fitness) * np.random.normal(0.0, 2.0, self.dim)
            I.x = np.clip(I.x, self.min_value, self.max_value)

        return group

    def evaluate_value(self):
        results = []
        for I in self.survivor_group:
            comparison = np.random.choice([_ for _ in self.survivor_group if _ != I], size=self.M)

            I.value = 0
            for c in comparison:
                if I.fitness < c.fitness:
                    I.value += 1

            results.append(I)

        self.survivor_group = results

    def select(self):
        sorted_group = sorted(self.survivor_group, key=lambda x: x.value, reverse=True)
        self.individual_group = sorted_group[:self.N]

    def record_param(self):
        if not self.record:
            return None

        os.makedirs(self.d_path, exist_ok=True)
        filename = f'{self.d_path}/param.json'
        params = {
            'N': self.N,
            'M': self.M,
        }
        with open(filename, 'w') as f:
            json.dump(params, f)

    def record_results(self, i):
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
        for M in [1, 1e1, 1e2]:
            ep = EP()
            param = {
                'N': int(N),
                'M': int(M),
            }
            ep.set_param(param)
            process = Process(target=ep.run)
            process_list.append(process)
            process.start()
            # print(ep.result)

    for p in process_list:
        p.join()


if __name__ == '__main__':
    main()
