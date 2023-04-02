# -*- coding: utf-8 -*-
# @Time    : 2023/3/23 16:29
# @Author  : 纪冠州
# @File    : main.py
# @Software: PyCharm 
# @Comment : 编程作业1.3.2对称矩阵的特征值与特征向量


import numpy as np

if __name__ == "__main__":
    epsilon = 0.000000001
    n = 3
    B = np.random.rand(n, n)
    A = np.dot(B, B.T)
    x = np.random.rand(n, 1)
    feature_value, feature_vector = np.linalg.eig(A)
    while True:
        x_0 = x
        y = np.dot(A, x_0)
        x = (y/np.linalg.norm(y)).reshape(n, 1)
        delta = np.fabs(x - x_0)
        print("x_k = {} delta = {}".format(x_0.T.squeeze(), delta.T.squeeze()))
        if np.linalg.norm(delta) < epsilon:
            break
    print("u_1 = {}".format(feature_vector.T[0]))
