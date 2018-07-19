import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits() # 숫자 데이터들을 가져옴

X_train, y_train = digits.data, digits.target
print(X_train.shape) # 8 x 8 이미지를 1열로 바꾸고 그 데이터가 총 1797장이 있음
# (1797, 64)

print(y_train.shape) # 각 이미지에 대한 숫자(0~9 까지를 나타냄)
# (1797,)

digits_index = 9 # 보여줄 숫자의 이미지
svm_model = svm.SVC(gamma=0.0001, C=100) # 학습 모델 생성
svm_model.fit(X_train, y_train) # 훈련데이터 학습, y_train 데이터의 모양에 따라 classification의 결과가 달라짐
plt.imshow(digits.images[digits_index], cmap=plt.cm.gray_r, interpolation='nearest') # 숫자를 그림으로 보여줌
plt.show()
