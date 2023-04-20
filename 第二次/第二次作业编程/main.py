# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 21:45
# @Author  : 纪冠州
# @File    : main.py
# @Software: PyCharm 
# @Comment : 第二次作业-编程题

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search

n = 4
m = n
x0 = np.array([1.2, 1, 1, 1])


def r(x):
    res = np.zeros(m)
    for i in range(0, n, 2):
        res[i] = 10 * (x[i + 1] - x[i] ** 2)
        res[i + 1] = 1 - x[i]
    return res


def f(x):
    return np.sum(r(x) ** 2)


def df(x):
    J = np.zeros((m, n))
    for i in range(0, n, 2):
        J[i, i] = -20 * x[i]
        J[i, i + 1] = 10
        J[i + 1, i] = -1
    return 2 * np.dot(J.T, r(x))


class SR1:
    def __init__(self, n):
        self.Hk = np.eye(n)

    def update(self, sk, yk):
        self.Hk = self.Hk + np.linalg.norm(sk - np.dot(self.Hk, yk)) / np.dot((sk - np.dot(self.Hk, yk)).T, yk)

    def dot(self, gk):
        return np.dot(self.Hk, gk)


class DFP:
    def __init__(self, n):
        self.Hk = np.eye(n)

    def update(self, sk, yk):
        self.Hk = self.Hk - np.linalg.norm(np.dot(self.Hk, yk)) / np.dot(np.dot(yk.T, self.Hk), yk) + np.linalg.norm(
            sk) / np.dot(yk.T, sk)

    def dot(self, gk):
        return np.dot(self.Hk, gk)


class BFGS:
    def __init__(self, n):
        self.Hk = np.eye(n)

    def update(self, sk, yk):
        pk = 1 / np.dot(sk.T, yk)
        self.Hk = np.dot(np.dot((np.eye(n) - np.dot(np.dot(pk, yk), sk.T)).T, self.Hk),
                         np.eye(n) - np.dot(np.dot(pk, yk), sk.T)) + np.dot(np.dot(pk, sk), sk.T)

    def dot(self, gk):
        return np.dot(self.Hk, gk)


def optimize_with_method(method):
    x = x0
    results = [(x, f(x))]
    alpha = 1
    while True:
        g = df(x)
        d = -method.dot(g)
        alpha_k = line_search(f=f, myfprime=df, xk=x, pk=d, c2=0.9)[0]
        if alpha_k == None:
            break
        elif isinstance(alpha_k, float):
            alpha = alpha_k
        else:
            alpha = alpha_k.squeeze()
        print(alpha, end="")
        s = alpha * d
        y = df(x + s) - g
        method.update(s, y)
        x = x + s
        f_value = f(x)
        print(x, f_value)
        results.append((x, f_value))
        if np.linalg.norm(f_value) < 1e-8:
            break
    return results


methods = [SR1, DFP, BFGS]

for method in methods:
    results_m = optimize_with_method(method(n))
    print(method.__name__, ':', len(results_m))
    plt.plot([r[1] for r in results_m], label=method.__name__)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective value')
    plt.show()
