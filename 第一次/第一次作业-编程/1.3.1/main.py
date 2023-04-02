# -*- coding: utf-8 -*-
# @Time    : 2023/3/22 22:00
# @Author  : 纪冠州
# @File    : main.py
# @Software: PyCharm 
# @Comment : 编程作业1.3.1使用线搜索确定步长和固定步长的梯度法

import numpy as np
from scipy.optimize import line_search
import matplotlib.pyplot as plot


def make_graph(l_value):
    num = len(l_value)
    loss = []
    for each in l_value:
        loss.append(float(np.fabs(each[1] - l_value[num-1][1])))
    plot.figure(figsize=[10, 6])
    plot.plot(loss)
    plot.xlabel("iteration", fontsize=12)
    plot.ylabel("Loss: $|f(x_k) - f(x^*)|$", fontsize=12)
    plot.yscale("log")
    plot.show()


class Optimize:

    def __init__(self, Q_0, b_0, c_0, x_0, epsilon=0.00000001):
        # 精度
        self.epsilon = epsilon
        # 二次函数参数
        self.Q = np.array(Q_0, dtype="float64")
        self.b = np.array(b_0, dtype="float64").reshape([-1, 1])
        self.c = c_0
        # 二次函数及梯度函数
        self.func = lambda x: 0.5 * np.dot(np.dot(x.T, self.Q), x).squeeze() + np.dot(self.b.T, x).squeeze() + self.c
        self.gradient = lambda x: np.dot(self.Q, x) + self.b
        self.gradient_2 = lambda x: np.reshape(np.dot(self.Q, x) + self.b, [1, -1])
        # 初始点
        self.x = np.array(x_0, dtype="float64").reshape([-1, 1])

    # 精确线搜索
    def extract_line_search(self):
        X = self.x
        value = []
        while True:
            f_value = self.func(X)
            f_gradient = self.gradient(X)
            if not (f_gradient == np.zeros(X.shape)).all():
                alpha = np.dot(f_gradient.T, f_gradient).squeeze() / np.dot(np.dot(f_gradient.T, self.Q),
                                                                            f_gradient).squeeze()
            else:
                alpha = 0
            # print("x = {} f = {} ∇f = {} α = {}".format(X.reshape([1, -1]).squeeze(), f_value, f_gradient.reshape([1, -1]).squeeze(), alpha))
            value.append([X.reshape([1, -1]).squeeze(), f_value, f_gradient.reshape([1, -1]).squeeze(), alpha])
            if np.linalg.norm(f_gradient) < self.epsilon:
                break
            X = X + np.dot(alpha, -f_gradient)
        make_graph(value)
        return value

    # 固定步长的梯度下降法
    def fixed_step_size_gradient_descent(self, fixed_alpha=0.01):
        X = self.x
        value = []
        while True:
            f_value = self.func(X)
            f_gradient = self.gradient(X)
            # print("x = {} f = {} ∇f = {} α = {}".format(X.reshape([1, -1]).squeeze(), f_value, f_gradient.reshape([1, -1]).squeeze(), fixed_alpha))
            value.append([X.reshape([1, -1]).squeeze(), f_value, f_gradient.reshape([1, -1]).squeeze(), fixed_alpha])
            if np.linalg.norm(f_gradient) < self.epsilon:
                break
            X = X - np.dot(fixed_alpha, f_gradient)
        make_graph(value)
        return value

    # 非精确线搜索_Wolfe准则
    def non_exact_line_search(self):
        X = self.x
        value = []
        while True:
            f_value = self.func(X)
            f_gradient = self.gradient(X)
            alpha = line_search(self.func, self.gradient_2, X, -f_gradient)[0]
            alpha = 0 if alpha is None else (alpha if isinstance(alpha, float) else alpha.squeeze())
            # print("x = {} f = {} ∇f = {} α = {}".format(X.reshape([1, -1]).squeeze(), f_value, f_gradient.reshape([1, -1]).squeeze(), alpha))
            value.append([X.reshape([1, -1]).squeeze(), f_value, f_gradient.reshape([1, -1]).squeeze(), alpha])
            if alpha == 0 or np.linalg.norm(f_gradient) < self.epsilon:
                break
            X = X - np.dot(alpha, f_gradient)
        make_graph(value)
        return value


if __name__ == "__main__":
    # 二次函数参数
    Q = [[1, 0], [0, 10]]
    b = [0, 0]
    c = 10
    # 初始点
    x_initial = [6, 6]

    opt = Optimize(Q, b, c, x_initial)
    opt.extract_line_search()
    opt.fixed_step_size_gradient_descent()
    opt.non_exact_line_search()
