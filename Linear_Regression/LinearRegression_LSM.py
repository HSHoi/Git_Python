# 선형 회귀 분석
import matplotlib.pyplot as plt
import numpy as np


# Hypothesis
def predict(x):
    return w0 + w1 * x


sample_data = [[10, 25], [20, 45], [30, 65], [50, 105]]  # sample data
X_train = []
y_train = []
X_train_a = []
y_train_a = []
total_size = 0
sum_xy = 0
sum_x = 0
sum_y = 0
sum_x_square = 0

for row in sample_data:
    X_train = row[0]  # row x 데이터
    y_train = row[1]  # row y 데이터
    X_train_a.append(row[0])  # 데이터를 보여주기 위해
    y_train_a.append(row[1])  # 데이터를 보여주기 위해
    sum_xy += X_train * y_train  # sigma x*y
    sum_x += X_train  # sigma x
    sum_y += y_train  # sigma y
    sum_x_square += X_train * X_train  # sigma x*x
    total_size += 1

w1 = (total_size * sum_xy - sum_x * sum_y) \
     / (total_size * sum_x_square - sum_x * sum_x)  # 선형회귀 분석 1차 식
w0 = (sum_x_square * sum_y - sum_xy * sum_x) \
     / (total_size * sum_x_square - sum_x * sum_x)  # 선형회귀 분석 1차 식

X_test = 40
y_predict = predict(X_test)

print(" w1 : ", w1)
# w1 :  4.571428571428571

print(" w0 : ", w0)
# w0 :  -40.714285714285715

print(" 예상 값 :", " x 값 :", X_test, " y_predict :", y_predict)
# 예상 값 :  x 값 : 40  t y_predict : 142.1428571428571

# 그래프 그리기

x_new = np.arange(0, 51)
y_new = predict(x_new)

plt.scatter(X_train_a, y_train_a, label='data')  # scatter 함수는 점으로 표시(산점도)
plt.scatter(X_test, y_predict, label='predict')
plt.plot(x_new, y_new, 'r', label='regression')  # plot 함수는 직선으로 표시
plt.xlabel("House Size")  # x 값 이름
plt.ylabel("House Price")  # y 값 이름
plt.title("Linear Regression")  # 그래프 제목
plt.legend()  # 범례 생성
plt.show()



