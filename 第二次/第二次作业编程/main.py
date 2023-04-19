# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 21:45
# @Author  : 纪冠州
# @File    : main.py
# @Software: PyCharm 
# @Comment : 第二次作业-编程题

import numpy as np
import matplotlib.pyplot as plt

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


def line_search_wolfe(fun, x0, g, alpha=1, c1=1e-4, c2=0.9):
    phi0 = fun(x0)
    dphi0 = np.dot(df(x0), alpha)
    phi = lambda a: fun(x0 + np.dot(a, alpha))
    dphi = lambda a: np.dot(df(x0 + np.dot(a, alpha)), alpha)
    a = 0
    b = np.inf
    while True:
        if phi(alpha).all() > (phi0 + c1 * alpha * dphi0).all() or (phi(alpha).all() >= phi(a).all() and a != 0):
            b = alpha
            alpha = (a + b) / 2
        else:
            if abs(dphi(alpha)).all() <= (-c2 * dphi0).all():
                break
            if dphi(alpha).all() >= 0:
                b = alpha
                alpha = (a + b) / 2
            else:
                a = alpha
                if b == np.inf:
                    alpha = 2 * a
                else:
                    alpha = (a + b) / 2
    return alpha


class SR1:
    def __init__(self, n):
        self.Hk = np.eye(n)

    def update(self, sk, yk):
        yksk = yk - np.dot(self.Hk, sk)
        yksk_norm = np.dot(yksk, sk)
        if yksk_norm > 0:
            self.Hk += np.outer(yksk, yksk) / yksk_norm

    def dot(self, gk):
        return np.dot(self.Hk, gk)


class DFP:
    def __init__(self, n):
        self.Hk = np.eye(n)

    def update(self, sk, yk):
        if np.dot(yk, sk) > 0:
            self.Hk = self.Hk - np.outer(np.dot(self.Hk, sk), np.dot(self.Hk, sk)) / np.dot(np.dot(sk, self.Hk), sk) \
                      + np.outer(yk, yk) / np.dot(yk, sk)

    def dot(self, gk):
        return np.dot(self.Hk, gk)


class BFGS:
    def __init__(self, n):
        self.Hk = np.eye(n)

    def update(self, sk, yk):
        yksk = yk - np.dot(self.Hk, sk)
        yksk_norm = np.dot(yksk, sk)
        if yksk_norm > 0:
            self.Hk += np.outer(yksk, yksk) / yksk_norm \
                       - (np.outer(np.dot(self.Hk, sk), np.dot(self.Hk, sk)) / np.dot(sk, np.dot(self.Hk, sk)))

    def dot(self, gk):
        return np.dot(self.Hk, gk)


def optimize_with_method(method):
    x = x0
    alpha = 1
    results = [(x, f(x))]
    #for i in range(100):
    while True:
        g = df(x)
        p = -method.dot(g)
        alpha = line_search_wolfe(f, x, g, alpha=alpha)
        s = alpha * p
        y = df(x + s) - g
        method.update(s, y)
        x = x + s
        results.append((x, f(x)))
        if np.linalg.norm(s) < 1e-8:
            break
    return results


results_sr1 = optimize_with_method(SR1(n))
results_dfp = optimize_with_method(DFP(n))
results_bfgs = optimize_with_method(BFGS(n))
print("SR1:"+str(len(results_sr1))+",DFP:"+str(len(results_dfp))+",BFGS:"+str(len(results_bfgs)))

plt.figure(1)
plt.plot([r[1] for r in results_sr1], label='SR1')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.figure(2)
plt.plot([r[1] for r in results_dfp], label='DFP')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.figure(3)
plt.plot([r[1] for r in results_bfgs], label='BFGS')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.show()
