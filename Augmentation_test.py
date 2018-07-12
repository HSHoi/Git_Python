import statsmodels.api as sm
import numpy as np

X1 = np.array([[10], [20], [30], [50]])
print(X1)
# [[10]
#  [20]
#  [30]
#  [50]]

# 오그멘테이션
X_train = sm.add_constant(X1)  # 주어진 행렬의 열을 추가
print(X_train)
# [[ 1. 10.]
#  [ 1. 20.]
#  [ 1. 30.]
#  [ 1. 50.]]

print(X_train.shape)
# (4, 2)

print(np.ones((X1.shape[0], 1)))  # (4 x 1) 행렬, 값이 1.0
# [[1.]
#  [1.]
#  [1.]
#  [1.]]

# np.hstack - 가로 쌓기, np.vstack - 세로 쌓기
X_train1 = np.hstack([np.ones((X1.shape[0], 1)), X1])  # 행렬 오른쪽에 데이터 추가(열 확장)
print(X_train1)
# [[ 1. 10.]
#  [ 1. 20.]
#  [ 1. 30.]
#  [ 1. 50.]]
