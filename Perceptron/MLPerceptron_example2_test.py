from sklearn.neural_network import MLPClassifier

X_train = [[0., 0.], [1., 1.]]
y_train = [[0, 1], [1, 1]]

'''
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
'''
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)  # 2-5-2-2
mlp.fit(X_train, y_train)

print(mlp.predict([[1., 2.]]))
# [[1 1]]

print(mlp.predict([[0., 0.]]))
# [[0 1]]

print("mlp.coefs_ :", mlp.coefs_)
# mlp.coefs_ :
# [array([[-0.15011367, -0.62860541, -0.90433213, -3.45938109, -0.63904618],
#        [-0.73749132, -1.5947694 , -0.2793927 , -3.28854097,  0.0702225 ]]),
# array([[ 0.30838904, -0.14960207],
#        [ 3.14928608, -0.65056811],
#        [-0.54615798,  0.54407041],
#        [ 4.36386369, -0.33753023],
#        [ 0.34792663,  0.68091737]]),
# array([[-3.58233912,  2.68515229],
#        [ 0.9049651 , -0.96123048]])]

print("coef.shape :", [coef.shape for coef in mlp.coefs_])
# coef.shape : [(2, 5), (5, 2), (2, 2)]

print("mlp.n_outputs_ : ", mlp.n_outputs_)  # 마지막 노드의 개수
# mlp.n_outputs_ :  2

print("mlp.classes_:", mlp.classes_)
# mlp.classes_: [0 1]

# softmax
print("clf.predict_proba([[2., 2.], [1., 2.]]) :", mlp.predict_proba([[2., 2.], [1., 2.]]))  # 소프트 맥스함수의 결과
# clf.predict_proba([[2., 2.], [1., 2.]] : [[0.99995432 0.99999835] [0.99995432 0.99999835]]

print("n_outputs_ : ", mlp.n_outputs_)  # 출력 노드의 수
# n_outputs_ : 2
