# 선형 회귀 분석 행렬(numpy) 사용해서 계산
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np


def predict(x):
    return w0 + w1 * x


X1 = np.array([[10], [20], [30], [50]])
y_label = np.array([[25], [45], [65], [105]])
X_train = sm.add_constant(X1)
print(X_train)  # (4 x 2)
# [[ 1. 10.]
#  [ 1. 20.]
#  [ 1. 30.]
#  [ 1. 50.]]

print(X_train.T)  # (2 x 4)
# [[ 1.  1.  1.  1.]
#  [10. 20. 30. 50.]]

# 선형회귀 분석 행렬 기반
w = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_label)
print(np.dot(X_train.T, X_train))  # (2 x 4) * (4 x 2) = (2 x 2)
# [[   4.  110.]
#  [ 110. 3900.]]

# 역행렬 계산(pseudo inverse)
print(np.linalg.inv(np.dot(X_train.T, X_train)))  # (2 x 2)
# [[ 1.11428571 -0.03142857]
#  [-0.03142857  0.00114286]]

# (2 x 2) * (2 x 4) = (2 x 4)
print(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T))  # (2 x 4)
# [[ 0.8         0.48571429  0.17142857 -0.45714286]
#  [-0.02       -0.00857143  0.00285714  0.02571429]]

# (2 x 4) * (4 x 1)
print(np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), y_label))  # (2 x 1)
# [[5.]
#  [2.]]

w0 = w[0]
w1 = w[1]

X_test = 40
y_predict = predict(X_test)
print("가중치 :", w1)
# w1 :  [2.]

print("상수 :", w0)
# w0 :  [5.]

print("예상 값 :", "x 값 :", X_test, "y_predict :", y_predict)
# 예상 값 :  x 값 : 40  predict(40) : [85.]

x_new = np.arange(0, 51) # 새로 데이터 생성
y_new = predict(x_new) # 만들어진 모델에 새로 생성한 데이터를 너보고 결과 생성

plt.scatter(X1, y_label, label="data")  # scatter 함수는 점으로 표시(산점도)
plt.scatter(X_test, y_predict, label="predict")
plt.plot(x_new, y_new, 'r-', label="regression")  # plot 함수는 직선으로 표시
plt.xlabel("House Size")  # x 값 이름
plt.ylabel("House Price")  # y 값 이름
plt.title("Linear Regression _ with numpy")  # 그래프 제목
plt.legend()  # 범례 생성
plt.show()  # 그래프 보기
