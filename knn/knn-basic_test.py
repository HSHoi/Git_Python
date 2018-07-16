import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Input data
X_train = np.array([[3.2, 3.1], [4.2, 4.2], [1.9, 6.5], [4.1, 5.0], [5.1, 6.9],
                    [2.3, 5.3], [3.2, 5.5], [3.5, 3.7], [4.5, 4.1], [3.4, 5.9],
                    [4.1, 3.5], [4.1, 5.7], [3.1, 4.2], [5.2, 4.2], [4.7, 6.5]])

# 주변을 탐색할 데이터 개수(가까운 거리 기준)
k = 3

# New_input_data
new_input_data = [4.3, 4.7]

# Plot input data
# plt.figure()
plt.subplot(2, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=100, color='k', label='Input Data')
plt.scatter(new_input_data[0], new_input_data[1], marker='o', s=100, color='r', label='New Input Data')
plt.legend()

# Build K Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
knn_model.fit(X_train)  # K-nn 연산을 하기 위해 사용(Y_train 값이 없음)

# 최 근접 이웃  인덱
distance, indices = knn_model.kneighbors([new_input_data])
print("distance : ", distance)  # new_input_data에 대한 가까운 점 3개와의 거리
# distance :  [[0.36055513 0.50990195 0.63245553]]

print("indices : ", indices)  # new_input_data에 대한 가까운 점 3개의 데이터 인덱스 번호
# indices :  [[3 1 8]]

# 값 확인
cnt = 0
for i in indices[0]:
    print("좌표 값 : {0}, 거리 : {1}, index : {2}".format(X_train[i], distance[0][cnt], i))
    cnt += 1
# 좌표 값 : [4.1 5. ], 거리 : 0.3605551275463989, index : 3
# 좌표 값 : [4.2 4.2], 거리 : 0.5099019513592784, index : 1
# 좌표 값 : [4.5 4.1], 거리 : 0.6324555320336764, index : 8

# Visualize the nearest neighbors along with the test datapoint
# plt.figure()
plt.subplot(2, 1, 2)
plt.title('{0} Nearest neighbors'.format(k))

plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=75, color='k', label='Input Data')
plt.scatter(X_train[indices][0][:][:, 0], X_train[indices][0][:][:, 1], marker='o', s=250, color='k', facecolors='none')
plt.scatter(new_input_data[0], new_input_data[1],
            marker='o', s=200, color='r', label='New Input Data')
# Print the 'k' nearest neighbors
print("\nK Nearest Neighbors:")

for rank, index in enumerate(indices[0][:k], start=1):
    print(str(rank) + " ==>", X_train[index])
    plt.arrow(new_input_data[0], new_input_data[1], X_train[index, 0] - new_input_data[0],
              X_train[index, 1] - new_input_data[1], head_width=0, fc='k', ec='k')

plt.legend()
plt.show()
