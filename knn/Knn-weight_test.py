# Generate sample data
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

np.random.seed(0)
X_train = np.sort(5 * np.random.rand(40, 1), axis=0) # 훈련 데이터 40개 생성
# print(X_train)

x_test = np.linspace(0, 5, 500)[:, np.newaxis] # 테스트 데이터 500개 생성
# print(x_test)

y_train = np.sin(X_train).ravel() # 샘플 데이터를 생성 40개
print(y_train)

# Add noise to targets
y_train[::5] += 1 * (0.5 - np.random.rand(8)) # 샘플 데이터에 노이즈 추가

# #############################################################################
# Fit regression model  
# 거리에 따른 가중치를 어떻게 줄것인가?
# uniform : 동일하게 , distance : 거리에 따른 가중치 조절
n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    predict = knn.fit(X_train, y_train).predict(x_test)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X_train, y_train, c='k', label='data')
    plt.plot(x_test, predict, c='g', label='Prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNN (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.show()
