import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)
print(a)
print('*' * 30)

b = np.array([[7, 8], [9, 10], [11, 12]])
print(b.shape)
print(b)
print('*' * 30)

# a와 b를 내적
c = np.dot(a, b)  # dot 함수가 내적
print('*' * 30)
print(c.shape)
print(c)

# 전치 행렬
a = np.array([[1, 2, 3], [4, 5, 6]])
print('*' * 30)
print(a)
at = a.T  # .T 가 전치 행렬
print(at)

# 역행렬
import numpy.linalg as lin

a = np.array([[1, 2], [4, 5]])
print('*' * 30)
print(a)

c = lin.inv(a)  # a의 역행렬
print(c.shape)
print(c)

e1 = np.dot(a, c)  # 단위행렬 나오는지 확인
print(e1)

x = np.array([1.0, 2.0, 3.0, 4.0])  # 1차원
print(x)
print(type(x))
print(x.shape)
