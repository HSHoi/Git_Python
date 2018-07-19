import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
from sklearn.neural_network import MLPClassifier


def plot_mlp(ppn):
    # plt.figure(figsize=(12, 8), dpi=60)
    #    model = Perceptron(n_iter=10, eta0=0.1, random_state=1).fit(X, y)
    model = ppn
    XX_min = X[:, 0].min() - 1;
    XX_max = X[:, 0].max() + 1;
    YY_min = X[:, 1].min() - 1;
    YY_max = X[:, 1].max() + 1;
    XX, YY = np.meshgrid(np.linspace(XX_min, XX_max, 1000), np.linspace(YY_min, YY_max, 1000))
    ZZ = model.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    cmap = matplotlib.colors.ListedColormap(sns.color_palette("Set3"))
    plt.contourf(XX, YY, ZZ, cmap=cmap)
    plt.scatter(x=X[y == 0, 0], y=X[y == 0, 1], s=200, linewidth=2, edgecolor='k', c='y', marker='^', label='0')
    plt.scatter(x=X[y == 1, 0], y=X[y == 1, 1], s=200, linewidth=2, edgecolor='k', c='r', marker='s', label='1')

    plt.xlim(XX_min, XX_max)
    plt.ylim(YY_min, YY_max)
    plt.grid(False)
    plt.xlabel("X1")
    plt.ylabel("X0")
    plt.legend()
    plt.show()


X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # 학습 데이터
# Labels

print(X.shape)

y = np.array([0, 1, 1, 0])  # target data

mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=[10, 10]).fit(X, y)  # 학습 모델 생성 및 학습 시작
plot_mlp(mlp)
print("학습 결과 : ", mlp.predict(X))
# 학습 결과 :  [0 1 1 0]

# 입력 층 +  출력층 +  히든층 :
print("신경망 깊이 :", mlp.n_layers_)
# 신경망 깊이 : 3

# MLP의 계층별 가중치 확인
print('len(mlp.coefs_) :', len(mlp.coefs_))
# len(mlp.coefs_) : 2


print("mlp.n_outputs_ : ", mlp.n_outputs_) # 출력 층의 계수
# mlp.n_outputs_ :  1

print("mlp.classes_:", mlp.classes_)  # 결과 데이터의 종류 0 아니면 1
# mlp.classes_: [0 1]

for i in range(len(mlp.coefs_)):
    number_neurons_in_layer = mlp.coefs_[i].shape[1] # 현재 층에있는 뉴런의 갯수
    print("number_neurons_in_layer :", number_neurons_in_layer, ' i : ', i)
    # number_neurons_in_layer : 4  i :  0

    # number_neurons_in_layer : 1  i :  1

    for j in range(number_neurons_in_layer):
        weights = mlp.coefs_[i][:, j]
        print(i, j, weights, end=", ")
        # 0 0 [ 0.31567168 -0.36902892],
        # 0 1 [0.26016766 0.59169499],
        # 0 2 [-3.43198216 -3.4475273 ],
        # 0 3 [3.86673773 3.86303838],

        # 1 0 [ 1.36996297  0.87485384 -5.04108867 -4.61923008],
        print()
    print()
