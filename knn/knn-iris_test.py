import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# import some data to play with
iris = datasets.load_iris()  # iris 데이터 획득

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X_train = iris.data[:, :2]  # 꽃받침 길이, 꽃받침 너비
# print(X_train)

x_train = iris.data # 총 150개의 데이터 [ 꽃받침 길이, 꽃받침 너비, 꽃잎의 길이, 꽃잎의 너비 ]
print(x_train.shape)
# (150, 4)

y_train = iris.target  # 꽃의 종류
# print(y_train)

weights = 'distance'
h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# we create an instance of Neighbours Classifier and fit the data.
k_list = [1, 10, 15, 100]
for k in k_list:
    # print(weights)

    # we create an instance of Neighbours Classifier and fit the data.
    knn_model = neighbors.KNeighborsClassifier(k, weights=weights)
    knn_model.fit(X_train, y_train)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1  # 꽃 받침 너비
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1  # 길이
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Put the result into a color plot
    plt.figure()

    # plt.subplot(2, 1, 1)

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')" % (k, weights))

plt.show()
