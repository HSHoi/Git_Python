import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0])  # 1차원
print(x)
# [1. 2. 3. 4.]
print(type(x))
# <class 'numpy.ndarray'>
print(x.shape)
# (4,)

x1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])  # 2차원
print(x1)
# [[1 2 3 4 5]
#  [6 7 8 9 0]]
print(x1.shape)
# (2, 5)

x2 = np.zeros(10)  # 1 x 10 1차원 배열 생성 후 0으로 초기화
print(x2)

x3 = np.zeros((3, 2))  # 3 x 2 2차원 배열 생성 후 0으로 초기화
print(x3)

x4 = np.ones(10)  # 1 x 10 1차원 배열 생성 후 1로 초기화
print(x4)

x5 = np.ones((3, 2))  # 3 x 2 2차원 배열 생성 후 1로 초기화
print(x5)

# array 사칙연산

x = np.array([[1, 2, 3], [4, 5, 6]])
x1 = x + x
print(x1)
# [[ 2  4  6]
#  [ 8 10 12]]

x2 = x1 - x
print(x2)
# [[1 2 3]
#  [4 5 6]]

x3 = x * x
print(x3)
# [[ 1  4  9]
#  [16 25 36]]

x4 = x3 / x
print(x4)
# [[1. 2. 3.]
# [4. 5. 6.]]

x = np.array([[2, 3], [4, 5]])
y = 5
x1 = x + y

print(type(y))
# <class 'int'>
print(type(x))
# <class 'numpy.ndarray'>
print(x1)
# [[ 7  8]
# [ 9 10]]

# array broadcasting

x2 = np.array([[2, 3], [7, 8]])
y2 = np.array([[5, 10]])
x3 = x2 + y2

print(x2.shape)
# (2, 2)
print(y2.shape)
# (1, 2)
print(x3.shape)
# (2, 2)
print(x3)
# [[ 7 13]
#  [12 18]]

# array slice

x1 = list(range(10))
x2 = x1[0:3]  # list 형 슬라이스
print(x2)
# [0, 1, 2]

x2[1] = 100
print(x2)
# [0, 100, 2]
print(x1)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

y1 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(y1)
# [0 1 2 3 4 5 6 7 8 9]

y2 = y1[0:3]
print(y2)
# [0 1 2]

y2[1] = 100 # numpy는 포인터 개념이기 때문에 원본 y1이 바뀜
print(y2)
# [  0 100   2]

print(y1) # y1 변경됨
# [  0 100   2   3   4   5   6   7   8   9]
