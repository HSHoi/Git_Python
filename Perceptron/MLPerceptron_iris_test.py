"""
Created on Wed Nov 29 21:14:47 2017
꽃받침의 너비와 길이에 따른 분류 확인
iris[0,1]

•‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
•‘sgd’ refers to stochastic gradient descent.
•‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score.
For small datasets, however, ‘lbfgs’ can converge faster and perform better.
"""

from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import confusion_matrix


def plot_mlp(model, title):
    XX_min = X[:, 0].min() - 1;
    XX_max = X[:, 0].max() + 1;
    YY_min = X[:, 1].min() - 1;
    YY_max = X[:, 1].max() + 1;
    XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000), np.linspace(YY_min, YY_max, 1000))
    ZZ = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    cmap = matplotlib.colors.ListedColormap(sns.color_palette("Set3"))

    plt.figure()

    plt.contourf(XX, YY, ZZ, cmap=cmap)
    plt.scatter(x=X[y == 0, 0], y=X[y == 0, 1], s=100, linewidth=2, edgecolor='k', c='y', marker='^', label='0')
    plt.scatter(x=X[y == 1, 0], y=X[y == 1, 1], s=100, linewidth=2, edgecolor='k', c='r', marker='s', label='1')
    plt.scatter(x=X[y == 2, 0], y=X[y == 2, 1], s=100, linewidth=2, edgecolor='k', c='b', marker='X', label='2')

    plt.xlim(XX_min, XX_max)
    plt.ylim(YY_min, YY_max)
    plt.grid(False)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Widht")
    plt.title(
        "{0} \n Training set Accuracy : {1} \n Test set Accuracy: {2}".format(title, model.score(X_train, y_train),
                                                                              model.score(X_test, y_test)))

    #    plt.title(title )
    plt.legend()

    print("훈련 세트 정확도: {0}".format(model.score(X_train, y_train)))
    print("테스트 세트 정확도: {0}".format(model.score(X_test, y_test)))
    print(confusion_matrix(y, model.predict(X)))


# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset  꽃잎의 너비, 길이
X = iris.data[:, :2]  # train data
print(X.shape)
# (150, 2)

y = iris.target  # target data
print(y.shape)
# (150,)

# train data, test data, target data 로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
print(X_train.shape)
# (135, 2)

print(X_test.shape)
# (15, 2)

print(y_train.shape)
# (135,)

print(y_test.shape)
# (15,)

# 디폴트 생성
mlp = MLPClassifier(solver='lbfgs', random_state=0)
mlp.fit(X_train, y_train)
title = 'hidden layer :100, solver= lbfgs'
# print("훈련 세트 정확도: {0}".format(mlp.score(X_train, y_train)))
# print("테스트 세트 정확도: {0}".format(mlp.score(X_test, y_test)))
# print(confusion_matrix(y, mlp.predict(X)))
plot_mlp(mlp, title)
print("mlp.n_outputs_ : ", mlp.n_outputs_)
print("mlp.classes_:", mlp.classes_)

# 1 개의 은닉층, 20 개의 노드
mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=(20))
mlp.fit(X_train, y_train)
title = 'hidden layer :20, solver= lbfgs'
# print("훈련 세트 정확도: {0}".format(mlp.score(X_train, y_train)))
# print("테스트 세트 정확도: {0}".format(mlp.score(X_test, y_test)))
# print(confusion_matrix(y, mlp.predict(X)))
plot_mlp(mlp, title)

# 20개 노드로 구성된 은닉층 2개
mlp = MLPClassifier(solver='lbfgs', random_state=0,
                    hidden_layer_sizes=(20, 20))
mlp.fit(X_train, y_train)
title = 'hidden layer :20  x 2 , solver= lbfgs'
# print("훈련 세트 정확도: {0}".format(mlp.score(X_train, y_train)))
# print("테스트 세트 정확도: {0}".format(mlp.score(X_test, y_test)))
# print(confusion_matrix(y, mlp.predict(X)))
plot_mlp(mlp, title)

# tanh 활성화 함수가 적용된 20 유닛으로 된 두 개의 은닉층
mlp = MLPClassifier(solver='lbfgs', activation='tanh',
                    random_state=0, hidden_layer_sizes=(20, 20))
mlp.fit(X_train, y_train)
title = 'hidden layer :100, solver= lbfgs, activation=tanh'

# print("훈련 세트 정확도: {0}".format(mlp.score(X_train, y_train)))
# print("테스트 세트 정확도: {0}".format(mlp.score(X_test, y_test)))
# print(confusion_matrix(y, mlp.predict(X)))
plot_mlp(mlp, title)

plt.show()
