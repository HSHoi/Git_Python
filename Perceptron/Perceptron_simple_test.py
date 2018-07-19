import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

colormap = np.array(['r', 'k'])


def make_cal(X_train, y_train):
    # Create the model
    net = Perceptron(n_iter=100,  # 내부적으로 100번 학습
                     verbose=0, random_state=None,
                     fit_intercept=True,  # 절편(bias) 값을 줄 것인지 안 줄것인지 설정
                     eta0=0.002)
    net.fit(X_train, y_train)  # Learning model

    # Print the results
    print("Prediction :", str(net.predict(X_train)))
    print("Actual     :" + str(y_train))
    print("Accuracy   :", str(net.score(X_train, y_train)))

    # Plot the original data
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colormap[y_train], s=40)
    # plt.scatter(X_train[:,0], X_train[:,1], c=colormap[y_train], s=40)

    # Output the values
    print('Coefficient 0:', net.coef_[0, 0])
    print('Coefficient 1:', net.coef_[0, 1])
    print('Bias : ', net.intercept_)

    # Calc the hyperplane (decision boundary)
    plt.ylim([0, 10])
    ymin, ymax = plt.ylim()

    print('ymin :', ymin, 'ymax :', ymax)

    w = net.coef_[0]
    a = -w[1] / w[0]
    xx = np.linspace(ymin, ymax)
    yy = a * xx - (net.intercept_[0]) / w[0]

    # Plot the line
    plt.plot(yy, xx, 'k-')
    plt.show()


# train_data set
X_train = np.array([[2, 2], [1, 3], [2, 3], [5, 3], [7, 3], [2, 4],
                    [3, 4], [6, 4], [1, 5], [2, 5], [5, 5], [4, 6], [6, 6], [5, 9]])

# target_data
y_train = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])

make_cal(X_train, y_train)
# Prediction : [0 0 0 1 1 0 0 1 0 0 1 1 1 1]
# Actual     :[0 0 0 1 1 0 0 1 0 0 1 1 1 1]
# Accuracy   : 1.0
# Coefficient 0: 0.03399999999999999
# Coefficient 1: -0.011999999999999976
# Bias :  [-0.058]
# ymin : 0.0 ymax : 10.0
