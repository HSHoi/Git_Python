import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs

# we create 40 separable points
X_train, y_train = make_blobs(n_samples=40, centers=2, random_state=6)  # 훈련 데이터와 target 값 생성
print(X_train.shape)
# (40, 2)

print(y_train.shape)
# (40,)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X_train, y_train)  # 훈련데이터 학습
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)  # 훈련 데이터 그래프 표시

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
print('XX :', XX.shape)
# XX : (30, 30)

xy = np.vstack([XX.ravel(), YY.ravel()]).T
print('xy :', xy)
print('xy.shape ', xy.shape, 'XX.shape', XX.shape)
# xy : [[  3.9763743  -11.07662259]
#  [  3.9763743  -10.73212865]
#  [  3.9763743  -10.38763471]
#  ...
#  [ 10.80436835  -1.77528615]
#  [ 10.80436835  -1.4307922 ]
#  [ 10.80436835  -1.08629826]]
# xy.shape  (900, 2) XX.shape (30, 30)

Z = clf.decision_function(xy).reshape(XX.shape)
print('Z shape :', Z.shape)
# Z shape : (30, 30)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=10, linewidth=3, facecolors='red')

plt.show()
