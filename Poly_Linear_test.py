import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


def make_nl_sample():
    np.random.seed(0)
    samples_number = 50
    X = np.sort(np.random.rand(samples_number))
    y = np.sin(2 * np.pi * X) + np.random.randn(samples_number) * 0.2
    X = X[:, np.newaxis]
    return (X, y)


X_train, y_train = make_nl_sample()

model = LinearRegression().fit(X_train, y_train)  # 선형 회귀 분석 모델 생성 및 학습
predict = model.predict(X_train)

# Polynomial regression
poly_linear_model = LinearRegression()  # LinearRegression 클래스 선언
n_degree = 3  # 다항식 회귀 분석의 차수 설정
polynomial = PolynomialFeatures(n_degree) # PolynomialFeatures 클래스 선언

# 훈련 데이터를 다항식 회귀 분석 모델에 맞게 변환
X_train_transformed = polynomial.fit_transform(X_train)  # 차수에 맞는 형태의 데이터 형태 변환 (1차원 배열을 4차원으로 변환)
print("X_train_transformed.shape :", X_train_transformed.shape)
# X_train_transformed.shape : (50, 4)

# 다항식 회귀 분석 학습
poly_linear_model.fit(X_train_transformed, y_train) # 훈련 데이터가 차서에 맞게 변형되었기 때문에 자동적으로 다항식 회귀 분석으로 설정 됨
pre2 = poly_linear_model.predict(X_train_transformed)  # 다항식 회귀 모델에 훈련 데이터 적용한 결과
linear_r2_score = r2_score(y_train, predict)
print(linear_r2_score)
# 0.47156863327552134

poly_r2_score = r2_score(y_train, pre2)
print(poly_r2_score)
# 0.9058799878291176

plt.scatter(X_train, y_train, label='Training Data')
plt.plot(X_train, predict, label='Linear Regression', color='r')
plt.plot(X_train, pre2, label='Poly Regression', color='b')
plt.legend()
# plt.title("Degree : {}\n linear_r2_score : {:.2e}\n poly_r2_score : {:.2e} ".format(n_degree, linear_r2_score, poly_r2_score))
plt.show()
