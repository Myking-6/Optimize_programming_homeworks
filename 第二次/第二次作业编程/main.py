# -*- coding: utf-8 -*-
# @Time    : 2023/4/18 21:45
# @Author  : 纪冠州
# @File    : main.py
# @Software: PyCharm 
# @Comment : 第二次作业-编程题

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_powell, fmin_bfgs, minimize, SR1

n = 4
m = n
x0 = np.array([1.2, 1, 1, 1])
x_t = np.array([1.0, 1.0, 1.0, 1.0])
ftol = 1e-8


def r(x):
    res = np.zeros(m)
    for i in range(0, n, 2):
        res[i] = 10 * (x[i + 1] - x[i] ** 2)
        res[i + 1] = 1 - x[i]
    return res


def f(x):
    return np.sum(r(x) ** 2)


f_t = f(x_t)


def df(x):
    J = np.zeros((m, n))
    for i in range(0, n, 2):
        J[i, i] = -20 * x[i]
        J[i, i + 1] = 10
        J[i + 1, i] = -1
    return 2 * np.dot(J.T, r(x))


sr1_losses = []


def getcallback(func, retall):
    def callback(xk, state=None):
        loss = np.abs(f_t - func(xk))
        if loss < ftol:
            return True
        else:
            if retall:
                sr1_losses.append(loss)
            return False

    return callback


minimum = minimize(fun=f, x0=x0, method="trust-constr", hess=SR1(), callback=getcallback(f, True))
dfp_minimum, dfp_retall = fmin_powell(func=f, x0=x0, retall=True, disp=False, callback=getcallback(f, False))
dfp_losses = []
for point in dfp_retall:
    dfp_losses.append(np.abs(f_t - f(point)))
bfgs_minimum, bfgs_retall = fmin_bfgs(f=f, x0=x0, retall=True, disp=False, callback=getcallback(f, False))
bfgs_losses = []
for point in bfgs_retall:
    bfgs_losses.append(np.abs(f_t - f(point)))

methods = {'SR1': sr1_losses, 'DFP': dfp_losses, 'BFGS': bfgs_losses}

for method in methods:
    print(method + ':' + str(len(methods[method])))
    plt.plot([r for r in methods[method]], label=method)
plt.yscale("log")
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Objective value')
plt.show()
