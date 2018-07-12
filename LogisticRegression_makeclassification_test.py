from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# sckit-learn 제공하는 셈풀 데이터 생성 함수
X_train, y_train = make_classification(n_samples=10, n_features=1, n_redundant=0, n_informative=1,
                                       n_clusters_per_class=1, random_state=4)  # 훈련 데이터 생성
# print(X_train)

model = LogisticRegression(C=1e5)
model.fit(X_train, y_train)
predict = model.predict(X_train)

# 계수 / 절편
print("model.coef_ :", model.coef_)
# model.coef_ : [[0.0020924]]

print("model.intercept_ :", model.intercept_)
# model.intercept_ : [0.25145146]

# 시그모이드 함수에 적용..
xx = np.linspace(-3, 3, 100)
sigm = sigmoid(model.coef_[0][0] * xx + model.intercept_[0])
precision, recall, thresholds = roc_curve(y_train, model.predict(X_train))

# 테스트 데이터 생성
# X_test = np.array([[-2], [2]])
# y_test = model.predict(X_test)  # 테스트 데이터의 결과

plt.plot(xx, sigm, label='sigmoid', c='red')
plt.scatter(X_train, y_train, marker='o', label='Training Data', s=100)
plt.scatter(X_train, predict, marker='x', label='Predict Data', c='b', s=200, lw=2, alpha=0.5)
# plt.scatter(X_test, y_test, marker='^', label='new_predict', c='green', s=100)
plt.xlim(-3, 3)  # x 데이터 범위 설정
plt.legend()
plt.show()
plt.plot(precision, recall)
plt.show()
