'''
output layer의 갯수를 확인하기 위한 프로그램

output layer가 1개인 프로그램


'''

from sklearn.neural_network import MLPClassifier

X_train = [[0., 0.], [1., 1.]]  # train data
y_test = [0, 1]  # target data

# multilayerperceptron model create
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,  # 일반화
                    hidden_layer_sizes=(5, 2),  # 레이어의 모양 2-5-2-1
                    random_state=1)

mlp.fit(X_train, y_test)  # learning

new_data = [[2., 2.], [-1., -2.]]  # test data

print("clf.predict([[2., 2.], [-1., -2.]] :", mlp.predict(new_data))  # test data 를 사용하였을 때의 결과
# clf.predict([[2., 2.], [-1., -2.]] : [1 0]

print("mlp.coefs_ :", mlp.coefs_)
# [array([[-0.14196276, -0.02104562, -0.85522848, -3.51355396, -0.60434709],
#        [-0.69744683, -0.9347486 , -0.26422217, -3.35199017,  0.06640954]]),
# array([[ 0.29164405, -0.14147894],
#        [ 2.39665167, -0.6152434 ],
#        [-0.51650256,  0.51452834],
#        [ 4.0186541 , -0.31920293],
#        [ 0.32903482,  0.64394475]]),
# array([[-4.53025854],
#        [-0.86285329]])]

print("coef.shape :", [coef.shape for coef in mlp.coefs_])
# coef.shape : [(2, 5), (5, 2), (2, 1)]

print("mlp.n_outputs_ : ", mlp.n_outputs_)
# mlp.n_outputs_ :  1

print("mlp.classes_:", mlp.classes_)
# mlp.classes_: [0 1]

# MLPClassifier supports only the Cross-Entropy loss function,
# which allows probability estimates by running the predict_proba method.
print("self._label_binarizer.y_type_ :", mlp._label_binarizer.y_type_)
# self._label_binarizer.y_type_ : binary

# softmax
print("clf.predict_proba([[2., 2.], [1., 2.]] :", mlp.predict_proba([[2., 2.], [1., 2.]])) # 소프트 맥스함수의 결과
# clf.predict_proba([[2., 2.], [1., 2.]] : [[1.96718015e-04 9.99803282e-01] [1.96718015e-04 9.99803282e-01]]

print("n_outputs_ : ", mlp.n_outputs_) # 출력 노드의 수
# n_outputs_ : 1
