import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()
# 활성화 함수 사용 이유
# 결과의 범위를 제한하고 계산의 편의성을 제공한다.

# Perceptron 모델을 직접 만들어 사용해 본다.
class Perceptron(object):
    def __init__(self, rate=0.01, epoch=1):
        self.rate = rate  # 학습률
        self.epoch = epoch  # 학습 횟수

    def fit(self, X, y):
        """Fit training data
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        """

        # weights   초기 [ 0. 0. 0.]   2개의 Feature(w1, w2), 1개의 바이어스의 웨이트값(w0).
        self.weight = np.zeros(1 + X.shape[1])  # 가중치(w) 를 계산을 위해 설정

        print("X.shape :", X.shape)
        # X.shape : (100, 2)

        print("weight.shape :", self.weight.shape)
        # weight.shape : (3,)

        print("weight :", self.weight)
        # weight : [0. 0. 0.]

        # Number of misclassifications
        self.errors = []  # Number of misclassifications

        # 퍼셉트론 학습 과정
        # 학습을 통해 w1~wi 까지의 웨이트 값을 계산한다.
        for i in range(self.epoch):
            err = 0
            for xi, target in zip(X, y):
                # target = 목표값, xi 는 입력 학습 데이터, xi.shape = (2, )
                # print('x1 : {0}, t : {1}'.format(xi, target))

                # Perceptron의 델타 규칙을 구현한다.
                # wi = wi + a*xi(t-y)
                delta_w = self.rate * (target - self.predict(xi))  # 퍼셉트론의 학습 공식 a(t-y)
                self.weight[1:] += delta_w * xi  # 퍼셉트론의 학습 공식 a(t-y)*xi
                self.weight[0] += delta_w
                print(self.weight[0])
                err += int(delta_w != 0.0)  # 0이 아닐경우 카운팅. 에러가 있는 없는지 확인을 위한 것
            self.errors.append(err)
        return self

    # predict 함수 사용 시 내적 계산을 수행 함.
    # test 데이터를 평가할 때 사용하는 방법(행렬 내적)
    def net_input(self, X):
        """Calculate net input"""
        #        print('X : ' ,  X, ' self.weight[1:] :', self.weight[1:], ' self.weight[0] :',  self.weight[0])
        # self.weight[0] == bias, np.dot(X, self.weight[1:]) == w1x1 + w2x2
        return np.dot(X, self.weight[1:]) + self.weight[0]

    # net_input 함수를 통해 내적된 결과를 1과 -1로 표현한다
    # 활성화 함수(계단 함수 사용)
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


# 0과 2 인덱스 값 사용  (100 X 2 )
X_train = df.iloc[0:100, [0, 2]].values
print(X_train.shape)
# (100, 2)

# 목표 값 설정 (100 X 1)
y_train = df.iloc[0:100, 4].values
print(y_train)
# ['Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa' 'Iris-setosa'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor' 'Iris-versicolor'
#  'Iris-versicolor' 'Iris-versicolor']

# y_train가 Iris-setosa 이면, -1 아니면 1
y_train = np.where(y_train == 'Iris-setosa', -1, 1)
print(y_train)
# [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
#  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
#  -1 -1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
#   1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
#   1  1  1  1]


plt.figure()
plt.scatter(X_train[:50, 0], X_train[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X_train[50:100, 0], X_train[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')
# plt.show()

plt.figure()
pn = Perceptron(0.1, 10)
pn.fit(X_train, y_train)
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
# plt.show()


plt.figure()
plot_decision_regions(X_train, y_train, classifier=pn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
