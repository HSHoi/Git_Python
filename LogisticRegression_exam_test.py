# 공부 시간과 시험 합격 여부에 대한 로지스틱 회귀 분석 모델
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


X_train = np.array([[10], [20], [100], [1000], [5000], [6000], [7000],
                    [8000], [9000], [10000], [10010], [10020], [14000]])
y_train = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression(C=1e5)
model.fit(X_train, y_train)
predict = model.predict(X_train)

print("model.coef_ :", model.coef_)
# model.coef_ : [[0.0020924]]

print("model.intercept_ :", model.intercept_)
# model.intercept_ : [-19.18871729]

# predict = model.predict(X_train)
xx = np.linspace(0, 15000, 100)  # 0 ~ 150000 사이에 균일하게 100개의 데이터 생성

# 선을 그리기위한 데이터(학습결과를 보여주기 위한 그래프)
sigm = sigmoid(model.coef_[0][0] * xx + model.intercept_[0]) # 함수에 맞춰 넣기위해 2차 리스트, 1차 리스트에서 값을 뻄
precision, recall, thresholds = roc_curve(y_train, model.predict(X_train))  # roc 커브 생성
print("preciosion : ", precision)
# preciosion :  [0. 1.]

print("recall : ", recall)
# recall :  [1. 1.]

print("threshold : ", thresholds)
# threshold :  [1 0]

# 새로운 테스트 데이터 생성
X_test = np.array([[8500], [12000]])
y_test = model.predict(X_test)  # 테스트 데이터의 결과

# 선형 모델
linear_model = LinearRegression().fit(X_train, y_train)
linear_predict = linear_model.predict(X_train)
# print("Logisticr 2_score :" , r2_score(y_train, predict))
# print("Linear   r2_score :" , r2_score(y_train, linear_predict))
# print(y_test)

plt.plot(xx, sigm, label='sigmoid', c='blue') # 로지스틱 회귀 분석 모델을 보여주기 위한 플랏
plt.plot(X_train, linear_predict, label='linear Regression', c='gray')
plt.scatter(X_train, y_train, marker='o', label='Training Data', s=100)
plt.scatter(X_train, predict, marker='x', label='Predict Data', c='red', s=100, lw=2, alpha=0.5)
plt.scatter(X_test, y_test, marker='^', label='new_predict', c='green', s=100)

plt.xlim(0, 15000)
plt.legend()
plt.show()

plt.ylim(0, 1.1)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.plot(precision, recall)

plt.show()
