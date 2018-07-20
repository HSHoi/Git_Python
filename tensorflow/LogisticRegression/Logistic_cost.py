import matplotlib.pyplot as plt
import math
import numpy as np

def log_a() :
    return 'A'

def log_b() :
    return 'B'


y = 0  # 0 또는 1의 값을 갖음

if y == 1 :
    print(log_a())
else :
    print(log_b())


if y == 0 :
    print(log_a())
else :
    print(log_b())


# if 문을 사용하지 않고 해결 :
print(y*log_a() + (1-y)*log_b())