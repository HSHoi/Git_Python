import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron


# Needed to show the plots inline
# %matplotlib inline
# Data
def and_cal(x):  # 0 이 있으면 무조건 0
    w1, w2, theta = 0.5, 0.5, 0.7
    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2

        # 계단 함수와 유사함
        if tmp <= theta:
            print(x1[0], 'AND', x1[1], ' = ', 0)
        elif tmp > theta:
            print(x1[0], 'AND', x1[1], ' = ', 1)


def or_cal(x):
    w1, w2, theta = 0.5, 0.5, 0.2
    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2
        if tmp <= theta:
            print(x1[0], 'OR', x1[1], ' = ', 0)
        elif tmp > theta:
            print(x1[0], 'OR', x1[1], ' = ', 1)


def nand_cal(x):  # 0 이 있으면 무조건 1
    w1, w2, theta = 0.5, 0.5, 0.5
    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2
        if tmp > theta:
            print(x1[0], 'NAND', x1[1], ' = ', 0)
        elif tmp <= theta:
            print(x1[0], 'NAND', x1[1], ' = ', 1)


# X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# and_cal(X_train)
# nand_cal(X_train)


def xor_cal(x):
    w1, w2, theta = 0.5, 0.5, 0.2
    for x1 in x:
        tmp = x1[0] * w1 + x1[1] * w2
        if tmp <= theta:
            print(x1[0], 'XOR', x1[1], ' = ', 0)
        elif tmp > theta:
            print(x1[0], 'XOR', x1[1], ' = ', 1)


def make_cal(X_train, y_train, lb):
    # Create the model
    net = Perceptron(n_iter=100,  # 내부적으로 100번 학습
                     verbose=0, random_state=None,
                     fit_intercept=True,  # 절편(bias) 값을 줄 것인지 안 줄것인지 설정
                     eta0=0.002)
    net.fit(X_train, y_train)

    # Print the results
    print("Prediction :", str(net.predict(X_train)))
    print("Actual     :" + str(y_train))
    print("Accuracy   :", str(net.score(X_train, y_train)))

    plt.figure()
    # Plot the original data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colormap[y_train], s=40)
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=colormap[y_train], s=40)
    plt.title("Perceptron [{0}] calculation  : ".format(lb))

    # Output the values
    print('Coefficient 0:', net.coef_[0, 0])
    print('Coefficient 1:', net.coef_[0, 1])
    print('Bias : ', net.intercept_)

    # Calc the hyperplane (decision boundary)
    ymin, ymax = plt.ylim()
    w = net.coef_[0]
    a = -w[1] / w[0]
    xx = np.linspace(ymin, ymax) # 직선 그려주기 위한 데이터 생성
    yy = a * xx - (net.intercept_[0]) / w[0]

    # Plot the line
    plt.plot(yy, xx, 'k-')
    plt.show()


colormap = np.array(['r', 'k'])

X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

# Labels
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])
y_nand = np.array([1, 1, 1, 0])
y_xor = np.array([0, 1, 1, 0])

make_cal(X, y_and, 'and')
# Prediction : [0 0 0 1]
# Actual     :[0 0 0 1]
# Accuracy   : 1.0
# Coefficient 0: 0.004
# Coefficient 1: 0.006
# Bias :  [-0.008]

make_cal(X, y_or, 'or')
# Prediction : [0 1 1 1]
# Actual     :[0 1 1 1]
# Accuracy   : 1.0
# Coefficient 0: 0.004
# Coefficient 1: 0.004
# Bias :  [-0.002]

make_cal(X, y_nand, 'nand')
# Prediction : [1 1 1 0]
# Actual     :[1 1 1 0]
# Accuracy   : 1.0
# Coefficient 0: -0.004
# Coefficient 1: -0.006
# Bias :  [0.008]

# make_cal(X, y_xor, 'xor')
plt.show()
