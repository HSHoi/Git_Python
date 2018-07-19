import matplotlib.pyplot as plt


# 평균제곱 오차를 구한다.
def cost(x, y, w):
    c = 0
    # x에 포함된 데이터 만큼 평균 제곱 오차를 구한다.
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) ** 2
    return c / len(x)


# X 범위(-2.0 ~ 5.0)에 대한 Cost 함수를 작성한다.
def cost_graph(X, y):
    #    X = [1, 2, 3]
    #    y = [1, 2, 3]
    # w와 cost의 그래프
    # cost = (wxi-yi)^2
    for i in range(-30, 50):
        w = i / 10
        c_result = cost(X, y, w)
        print(w, c_result)
        plt.plot(w, c_result, 'ro')
    plt.show()


# 경사하강법 (배치 경사 하강법)
def gredient_descent(x, y, w):
    delta = 0
    for i in range(len(x)):
        # Hypothesis
        hx = w * x[i]
        delta += (hx - y[i]) * x[i]
        print("delta : ", delta, " hx : ", hx, "y[", i, "] :", y[i], "x[", i, "] :", x[i])
    return delta / len(x)


def preidct(x, w):
    for i in range(len(x)):
        result = x[i] * w
        print('input ', x[i], 'predict :', result)


X = [1, 2, 3]
y = [1, 2, 3]
# X = [1, 2, 3, 4, 5]
# y = [1, 4, 6, 8, 10]
w = 10
a = 0.1  # 1일때 발산, 0.1로 테스트

for i in range(100):  # 학습 횟수 설정
    # Early Stop을 위하여 cost를 계산한다.
    c = cost(X, y, w)

    # Gredient Descent를 통해 가중치의 변위(Delta)를 구한다.
    g = gredient_descent(X, y, w)

    # 가중치 = 가중치 + 학습률 * 변위(Delta)     1 step w :   = 10 - 0.1 * 4.2 --> 5.8
    # 가중치 = 가중치 - 학습률 * 변위(Delta) 이것도 상관 없음
    w -= a * g
    print(" w :", w)

    # early Stopping.... 중간에 탈출할 수 있는 방법임...(다양한 방법으로 적용필요)
    if c < 1.0e-25:
        print("HERE!!")
        break
    print('Loop Count : [', i, ']', 'Cost: [', c, ']')

x = [6, 7]
preidct(x, w)

# Cost 함수를 시각화 해본다.
cost_graph(X, y)
