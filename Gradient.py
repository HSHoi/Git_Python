import numpy as np

# 미분  df(x) = lim f(x+h) - f(x) / h (h->0)
def getDifferential(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def equation1(x):
    return 10 * x ** 3 + 5 * x ** 2 + 4 * x

def equation2(x):
    return 0.01*x**2 + 0.1*x

print(getDifferential(equation1, 5))
print(getDifferential(equation2, 10))

# 편미분
def function_1(x):  # 평미분 됨
    return x*x + 4.0 **2.0

print(getDifferential(function_1, 3))

def function_2(y): #편미분 됨
    return 3.0**2.0 + y*y

print(getDifferential(function_2, 4.0))



# 식 f = x**2 + y**2
# 점 x=3, y=4에서의 편미분을 구하시오
