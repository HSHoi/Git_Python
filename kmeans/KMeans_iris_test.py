import numpy as np
import matplotlib.pyplot as plt
# Though the following import is not directly being used, it is required
# for 3D projection to work
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import silhouette_score

# np.random.seed(5)
iris = datasets.load_iris()  # iris data 불러오기
X_train = iris.data
y_train = iris.target

estimators = [('K_means 8', KMeans(n_clusters=8)), ('K-Means 3', KMeans(n_clusters=3))]

fignum = 1
titles = ['8 clusters', '3 clusters']

for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134) # 3차원 그래프 생성
    est.fit(X_train) # 학습 시작
    labels = est.labels_ # 분류된 각 데이터의 값

    print(name, ':', est.labels_)
    # K_means 8 : [1 1 1 1 1 6 1 1 1 1 6 1 1 1 6 6 6 1 6 6 6 6 1 1 1 1 1 6 1 1 1 6 6 6 1 1 6
    #  1 1 1 1 1 1 1 6 1 6 1 6 1 5 5 5 2 5 2 5 4 5 2 4 2 2 5 2 5 2 2 2 2 7 2 7 5
    #  5 5 5 5 5 4 4 4 2 7 2 5 5 2 2 2 2 5 2 4 2 2 2 5 4 2 0 7 3 0 0 3 2 3 0 3 0
    #  7 0 7 7 0 0 3 3 7 0 7 3 7 0 3 7 7 0 3 3 3 0 7 7 3 0 0 7 0 0 0 7 0 0 0 7 0
    #  0 7]
    # K-Means 3 : [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    #  1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
    #  2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 0 0 0 0 2 0 0 0 0
    #  0 0 2 2 0 0 0 0 2 0 2 0 2 0 0 2 2 0 0 0 0 0 2 0 0 0 0 2 0 0 0 2 0 0 0 2 0
    #  0 2]

    ax.scatter((X_train[:, 3]), X_train[:, 0], X_train[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

    # 실루엣 기법을 이용하여 K-mean 알고리즘 결과 평가
    # -1 <= sihoutte_score <= 1
    # 1에 가까울 수록 올바른 클러스터에 분류된 것
    print('Sihoutte_score : ', silhouette_score(X_train, est.labels_,
                                                metric='euclidean', sample_size=len(X_train)))
    # Sihoutte_score :  0.347319474764622
    # Sihoutte_score :  0.5525919445213676

# Plot the ground truth
fig = plt.figure(fignum, figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X_train[y_train == label, 3].mean(),
              X_train[y_train == label, 0].mean(),
              X_train[y_train == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
y = np.choose(y_train, [1, 2, 0]).astype(np.float)
ax.scatter(X_train[:, 3], X_train[:, 0], X_train[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')  # 3
ax.set_ylabel('Sepal length')  # 0
ax.set_zlabel('Petal length')  # 2
ax.set_title('Ground Truth')
ax.dist = 12

plt.show()
