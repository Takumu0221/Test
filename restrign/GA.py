import numpy as np



class Individual:
    def __init__(self,
                 dim = 50):
        self.x = np.zeros(dim)
        self.fitness = 0


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

    def init(self):
        # 個体の生成・初期化

