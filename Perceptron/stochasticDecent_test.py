import numpy as np
import matplotlib.pyplot as plt

# scikit-Learn, scipy, sklearn

# 총 100개의 임의의 데이터 더미값, X, y, 분류 값

action = np.loadtxt('action.txt', delimiter=',')


# print(action)

def action_graph(action):
    for _, x1, x2, c in action:
        # 점의 색을 if문으로 선택
        plt.plot(x1, x2, 'ro' if c else 'go')

    # plt.show()


def decision_boundary(w, c, lb):
    #    print("w :", w, " type(w) : " , type(w))
    #   상수 , 가중치1, 가중치2
    b, w1, w2 = w

    print("b :", b, " w1 :", w1, " w2 :", w2)
    # y = w1 * x1 + w2 * x2 + b   # 가 0이 되는 식이  중간 값 : 0 임 ==> 시그모이드 값 :0.5 를 구함
    # sigmoid의 e^ 0 의 계산 값은 1 임!!     1/(1 + 1) ==> 0.5
    # -(w1 * x1 + b)/w2 = x2      # x1 을 기준 값으로 하여 x2 좌표를 예측하여 그림으로 그리기 위해
    y1 = -(w1 * -4 + b) / w2
    y2 = -(w1 * 4 + b) / w2
    plt.plot([-4, 4], [y1, y2], c, label=lb)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#
def gradient_descent(x, y):
    m, n = x.shape  # 100 , 3
    w = 0

    for _ in range(m):  # 100 번

        # hypothesis 가 시그모이드 인 모델..
        w -= 0  # (3, 1) = (3,1)
    return w.reshape(-1)  # 사용의 편의를 위해  1열로 차원 변경


# 확률적 경사 하강법
def gradient_stochastic(x, y):
    m, n = x.shape  # 100, 3
    w = 0

    for i in range(m * 10):  # ~ 999 번의 미니배치를 수행한다.
        w -= 0  # (3,) = (3,)

    return w


# 확률적 경사 하강법
def gradient_stochastic_random(x, y):
    m, n = x.shape  # 100, 3
    w = 0

    for i in range(m * 10):
        w -= 0  # (3,) = (3,)

    return w


# 똑 같은 것만 수행하므로 성능이 개선 되지 않음
def gradient_minibatch(x, y):
    m, n = x.shape  # 100 , 3
    w = np.zeros([n, 1])  # (3 , 1)
    lr = 0.01
    epochs = 10  # 내가 가지고 있는 데이터 셋 전체를 몇번 사용할 것인지
    batch_size = 5  # 한번에 몇 개 처리할 것인가

    for _ in range(epochs):
        w = 0
    return w.reshape(-1)  # 사용의 편의를 위해  1열로 차원 변경


# 미니배치 성능 개선을 위해 epochs가 끝나면 섞음
def gradient_minibatch_random(x, y):
    m, n = x.shape  # 100 , 3
    w = np.zeros([n, 1])  # (3 , 1)
    lr = 0.01
    epochs = 10  # 내가 가지고 있는 데이터 셋 전체를 몇번 사용할 것인지
    batch_size = 5  # 한번에 몇 개 처리할 것인가

    for i in range(epochs):
        w += 0

        # 하나의 epochs가 끝날때 섞어줌...
        t = np.random.randint(1000)
        np.random.seed(t)
        np.random.shuffle(x)
        np.random.seed(t)
        np.random.shuffle(y)

        # np.random.seed(i)
        # np.random.shuffle(x)
        # np.random.seed(i)
        # np.random.shuffle(y)

    return w.reshape(-1)  # 사용의 편의를 위해  1열로 차원 변경


# 미니배치 성능 개선을 위해 epochs가 끝나면 섞음
def gradient_minibatch_shuffle(action):
    m, n = 100, 3  # 100 , 3
    w = np.zeros([3, 1])  # (3 , 1)
    lr = 0.01
    epochs = 10  # 내가 가지고 있는 데이터 셋 전체를 몇번 사용할 것인지
    batch_size = 5  # 한번에 몇 개 처리할 것인가

    for i in range(epochs):
        w += 0
        # 하나의 epochs가 끝날때 섞어줌...     단.  요소의 크기가 클 경우 메모리에 과부하가 생길 수 있으므로 --> 인덱스 배열을 만들어 인덱스의 셔플을 통해 데이터를 가져옴
        np.random.shuffle(action)

        # np.random.seed(i)
        # np.random.shuffle(x)
        # np.random.seed(i)
        # np.random.shuffle(y)

    return w.reshape(-1)  # 사용의 편의를 위해  1열로 차원 변경


# 데이터가 100만개 정도 되면 전체 다 할 수 없으므로.... 확률적으로 적용
action_graph(action)

#  상수열(1.0) , x1 , x2 , target
x = action[:, :-1]
y = action[:, -1:]
print('x :', x)
print('y :', y)
print(x.shape, y.shape)

# w = gradient_stochastic(x, y)

# w= gradient_descent(x, y)
# print(w)
import time


def show_elapsed(f, x, y):
    #    start = time.time()
    #    f(x, y)
    #    print('{}'.format(time.time() - start))
    start = time.clock()
    f(x, y)
    print('{}'.format(time.clock() - start))


decision_boundary(gradient_descent(x, y), 'r', 'gradient_descent')
decision_boundary(gradient_stochastic(x, y), 'g', 'gradient_stochastic')
decision_boundary(gradient_stochastic_random(x, y), 'b', 'gradient_stochastic_random')
decision_boundary(gradient_minibatch(x, y), 'y', 'gradient_minibatch')
decision_boundary(gradient_minibatch_random(x, y), 'k', 'gradient_minibatch_random')
decision_boundary(gradient_minibatch_shuffle(action), 'm', 'gradient_minibatch_shuffle')
plt.legend()
plt.show()

'''  # 시간 측정
show_elapsed(gradient_descent,x,y)
show_elapsed(gradient_stochastic,x,y)
show_elapsed(gradient_stochastic_random,x,y)
show_elapsed(gradient_minibatch,x,y)
show_elapsed(gradient_minibatch_random,x,y)
'''

# np.random.rand(3)
