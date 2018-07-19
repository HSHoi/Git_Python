# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 23:26:49 2017

 XOR 연산을 MLPClassifier 이용하여 해결.

@author: BEAST
"""

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns


def plot_mlp(ppn):
    # plt.figure(figsize=(12, 8), dpi=60)
    #    model = Perceptron(n_iter=10, eta0=0.1, random_state=1).fit(X, y)
    model = ppn
    XX_min = X[:, 0].min() - 1;
    XX_max = X[:, 0].max() + 1;
    YY_min = X[:, 1].min() - 1;
    YY_max = X[:, 1].max() + 1;
    XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000), np.linspace(YY_min, YY_max, 1000))
    ZZ = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    cmap = matplotlib.colors.ListedColormap(sns.color_palette("Set3"))
    plt.contourf(XX, YY, ZZ, cmap=cmap)
    plt.scatter(x=X[y == 0, 0], y=X[y == 0, 1], s=200, linewidth=2, edgecolor='k', c='y', marker='^', label='0')
    plt.scatter(x=X[y == 1, 0], y=X[y == 1, 1], s=200, linewidth=2, edgecolor='k', c='r', marker='s', label='1')

    plt.xlim(XX_min, XX_max)
    plt.ylim(YY_min, YY_max)
    plt.grid(False)
    plt.xlabel("X1")
    plt.ylabel("X0")
    plt.legend()
    plt.show()


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# Labels

print(X.shape)

y = np.array([0, 1, 1, 0])

# X, y = X_train, y_train
# solver =  {‘lbfgs’, ‘sgd’, ‘adam’},
# hidden Layer의 노드 갯수(뉴런 갯수)  :  10   4
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 4]).fit(X, y)
#mlp = MLPClassifier(solver='lbfgs', random_state=0).fit(X, y)
plot_mlp(mlp)
print("학습 결과 : " ,mlp.predict(X))
# 입력 층 +  출력층 +  히든층 :
print("신경망 깊이 :" , mlp.n_layers_)
# MLP의 계층별 가중치 확인
print('len(mlp.coefs_) :', len(mlp.coefs_))

print("mlp.n_outputs_ : ", mlp.n_outputs_)
print("mlp.classes_:", mlp.classes_)

for i in range(len(mlp.coefs_)):
    number_neurons_in_layer = mlp.coefs_[i].shape[1]
    print("number_neurons_in_layer :", number_neurons_in_layer, ' i : ', i)
    for j in range(number_neurons_in_layer):
        weights = mlp.coefs_[i][:, j]
        print(i, j, weights, end=", ")
        print()
    print()
