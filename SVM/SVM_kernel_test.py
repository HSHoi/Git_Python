import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)  # train data
y = np.sin(X).ravel()  # target data
print(y)
# [ 0.14415918  0.17796574  0.22978773  0.24928643  0.32014619  0.3629617
#   0.70427365  0.72169941  0.78309245  0.80656999  0.84330679  0.9218538
#   0.96352582  0.99939807  0.9366527   0.86787704  0.86145011  0.78439525
#   0.72848344  0.65509942  0.57479729  0.37470255  0.27513696  0.24822033
#   0.09237645  0.03293112  0.01079613 -0.06667189 -0.07494893 -0.22322095
#  -0.43931238 -0.54825618 -0.59995522 -0.85384305 -0.98249348 -0.98535229
#  -0.99667893 -0.99887435 -0.99247294 -0.96955196]


# #############################################################################
# Add noise to targets
# y[::5]는 1차원 배열에서 5배수 번째 인덱스에만 특정 랜덤 값을 더해줌
y[::5] += 3 * (0.5 - np.random.rand(8))
print(y)
# [ 0.04361009  0.17796574  0.22978773  0.24928643  0.32014619  0.13695542
#   0.70427365  0.72169941  0.78309245  0.80656999  0.3792032   0.9218538
#   0.96352582  0.99939807  0.9366527   1.22007951  0.86145011  0.78439525
#   0.72848344  0.65509942 -0.91410799  0.37470255  0.27513696  0.24822033
#   0.09237645  0.42416063  0.01079613 -0.06667189 -0.07494893 -0.22322095
#  -0.86223429 -0.54825618 -0.59995522 -0.85384305 -0.98249348 -2.24695741
#  -0.99667893 -0.99887435 -0.99247294 -0.96955196]

# #############################################################################
# Fit regression model
# support vector machine regression(kinds of regression)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=4)

y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
