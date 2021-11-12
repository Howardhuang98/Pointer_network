#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/10 14:35  
------------      
"""
import itertools
import math
import random
import tensorflow as tf
import numpy as np
from keras.utils.np_utils import to_categorical
from tqdm import tqdm


class Tsp:
    """
    简单的Tsp问题类，用于生成数据
    如果需要大量的数据请查看tsp_data，由Pointer network 作者提供
    http://goo.gl/NDcOIG
    """
    def next_batch(self, batch_size=1):
        X, Y = [], []
        for b in tqdm(range(batch_size)):
            points = self.generate_data()
            solved = self.solve_tsp_dynamic(points)
            X.append(points), Y.append(solved)
        return np.asarray(X), np.asarray(Y)

    def length(self, x, y):
        return (math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2))

    def solve_tsp_dynamic(self, points):
        # calc all lengths
        all_distances = [[self.length(x, y) for y in points] for x in points]
        # initial value - just distance from 0 to
        # every other point + keep the track of edges
        A = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1])
             for idx, dist in enumerate(all_distances[0][1:])}
        cnt = len(points)
        for m in range(2, cnt):
            B = {}
            for S in [frozenset(C) | {0}
                      for C in itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j],
                                      A[(S - {j}, k)][1] + [j])
                                     for k in S if k != 0 and k != j])
            A = B
        res = min([(A[d][0] + all_distances[0][d[1]], A[d][1])
                   for d in iter(A)])
        return res[1]

    def generate_data(self, N=10):
        radius = 1
        rangeX = (0, 10)
        rangeY = (0, 10)
        qty = N

        deltas = set()
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                if x * x + y * y <= radius * radius:
                    deltas.add((x, y))

        randPoints = []
        excluded = set()
        i = 0
        while i < qty:
            x = random.randrange(*rangeX)
            y = random.randrange(*rangeY)
            if (x, y) in excluded:
                continue
            randPoints.append((x, y))
            i += 1
            excluded.update((x + dx, y + dy) for (dx, dy) in deltas)
        return randPoints

    def cost(self, x, y):
        cost = 0
        for i in range(len(y[0]) - 1):
            point_idx = y[0][i]
            point_idx2 = y[0][i + 1]
            cost += self.length(x[0][point_idx], x[0][point_idx2])
        return cost


if __name__ == "__main__":
    # 准备数据
    p = Tsp()
    X, Y = p.next_batch(1000000)
    YY = []
    for y in Y:
        YY.append(to_categorical(y))
    YY = np.asarray(YY)
    np.save(r"data/X-1000000.npy", X)
    np.save(r"data/Y-1000000.npy", Y)
    np.save(r"data/YY-1000000.npy", YY)
