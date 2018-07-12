# 선형 회귀 분석 scikit_learn 라이브러리 사용
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


def predict(x):
    return w0 + w1 * x


X_train = np.array([[10], [20], [30], [50]])  # x 데이터
y_train = np.array([[25], [45], [65], [105]])  # y 데이터
model = LinearRegression(fit_intercept=True)  # 선형 회귀 분석 모델 생성
model.fit(X_train, y_train) # 선형 회귀 분석 계산
X_test = 40 # 테스트 데이터
y_predict = model.predict(X_test) # 테스트 데이터에 대한 결과 값
print(y_predict)
# [[85.]]

y_pred = model.predict(X_train)
print(y_pred)
# [[ 25.]
#  [ 45.]
#  [ 65.]
#  [105.]]

mse = mean_squared_error(y_train, y_pred)
print(mse)
# 9.150786500563737e-29

print(" 가중치 : ", model.coef_)
# 가중치 :  [[2.]]

print(" 상수 : ", model.intercept_)
# 상수 :  [5.]

print(" 예상 값 :", " x 값 :", X_test, " y_predict :", y_predict)
# 상수 :  [5.]

w1 = model.coef_
w0 = model.intercept_

x_new = np.arange(0, 51)
y_new1 = predict(x_new)
print(y_new1)
# [[  5.   7.   9.  11.  13.  15.  17.  19.  21.  23.  25.  27.  29.  31.
#    33.  35.  37.  39.  41.  43.  45.  47.  49.  51.  53.  55.  57.  59.
#    61.  63.  65.  67.  69.  71.  73.  75.  77.  79.  81.  83.  85.  87.
#    89.  91.  93.  95.  97.  99. 101. 103. 105.]]

y_new = y_new1.reshape(-1, 1)
print(y_new)
# [[  5.]
#  [  7.]
#  [  9.]
#  [ 11.]
#  [ 13.]
#  [ 15.]
#  [ 17.]
#  [ 19.]
#  [ 21.]
#  [ 23.]
#  [ 25.]
#  [ 27.]
#  [ 29.]
#  [ 31.]
#  [ 33.]
#  [ 35.]
#  [ 37.]
#  [ 39.]
#  [ 41.]
#  [ 43.]
#  [ 45.]
#  [ 47.]
#  [ 49.]
#  [ 51.]
#  [ 53.]
#  [ 55.]
#  [ 57.]
#  [ 59.]
#  [ 61.]
#  [ 63.]
#  [ 65.]
#  [ 67.]
#  [ 69.]
#  [ 71.]
#  [ 73.]
#  [ 75.]
#  [ 77.]
#  [ 79.]
#  [ 81.]
#  [ 83.]
#  [ 85.]
#  [ 87.]
#  [ 89.]
#  [ 91.]
#  [ 93.]
#  [ 95.]
#  [ 97.]
#  [ 99.]
#  [101.]
#  [103.]
#  [105.]]

plt.scatter(X_train, y_train, label="data")  # scatter 함수는 점으로 표시(산점도)
plt.plot(x_new, y_new, 'r-', label="regression")  # plot 함수는 직선으로 표시
plt.scatter(X_test, y_predict, label="predict")
plt.xlabel("x")  # x 값 이름
plt.ylabel("y")  # y 값 이름
plt.title("Linear Regression with sckit_learn")  # 그래프 제목
plt.legend()  # 범례 생성
plt.show()  # 그래프 보기
