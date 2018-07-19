import numpy as np
from sklearn.linear_model import Perceptron
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap


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
    # 학습된 신경망을 통해 컬러맵을 계산을 수행한다..
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


# Iris 데이터 읽어오기
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# df.tail()


# iris 데이터 인덱스가  4인 컬럼(0부터 시작) 100개 데이터 가져오기
y_train = df.iloc[0:100, 4].values
print('y_train :', y_train)

# y가 Iris-setosa 이면, -1 아니면 1
y_train = np.where(y_train == 'Iris-setosa', -1, 1)
print('y_train :', y_train)

# 0번째 2번째 컬럼 데이터를 X 데이터로 가져오기
X_train = df.iloc[0:100, [0, 2]].values
print(X_train)

plt.scatter(X_train[:50, 0], X_train[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X_train[50:100, 0], X_train[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')
plt.show()

pn = Perceptron(max_iter=10, eta0=0.1, random_state=0)
pn.fit(X_train, y_train)

print((y_train != pn.predict(X_train)).sum())
print("Error :", (y_train != pn.predict(X_train)).sum())
print("SCORE :", pn.score(X_train, y_train))

'''
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()
'''

plot_decision_regions(X_train, y_train, classifier=pn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_train, pn.predict(X_train)))
# Confusion Matrix
# Confusion Matrix 는 원 목표 값의 Class 와 머신 러닝 모델이 예측한 Class 가 일치하는 개수를 counting 한 결과를 제공
# Confusion Matrix 에서는 원 목표 값의 Class 정보를 행(row)으로, 모델의 예측에 의한 클래스를 열(Column)으로 Matrix 형태로 표현
#                     -1로 예측  1로 예측
# -1(원 데이터 Class) [    [50     0]
#  1(원 데이터 Class) [      0    50]]
