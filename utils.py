#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Contact :   huanghoward@foxmail.com
@Modify Time :    2021/11/10 14:35  
------------      
"""
import numpy as np


def cost(x, y):
    """
    用于计算
    :param x: shape = [batch, n, 2]
    :param y: shape = [batch, n, n]
    :return:
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("Error!")
    else:
        batch_size = x.shape[0]
        travels = []
        for i in range(batch_size):
            points = x[i]
            idx = y[i].argmax(axis=1)
            travel = 0
            for j in range(len(idx) - 1):
                city1 = idx[j]
                city2 = idx[j + 1]
                travel += np.sqrt(sum(np.power(points[city2] - points[city1], 2)))
            travels.append(travel)
        travels = np.asarray(travels)
        return travels

def prob2rank(probs):
    """

    :param probs: [n,n]
    :return: rank: [n]
    """
    idx = probs.argmax(axis=1)
    return idx



def draw(x, y):
    """
    一次只允许画一个样本
    :param x:[n,2]
    :param y:[n,n]
    :return:
    """
    pass


if __name__ == "__main__":
    x_test = np.load(r"tsp_data/tsp_5/tsp5_test_x.npy")
    y_test = np.load(r"tsp_data/tsp_5/tsp5_test_y.npy")
    print(cost(x_test, y_test))
