import tensorflow as tf

x = [1, 2, 3]  # train data
y = [1, 2, 3]  # traget data

# 가중치, 균등분포로 랜덤하게 숫자를 만들기 위함
w = tf.Variable(tf.random_uniform([1], -1, 1))  # 초기값을 균등하게 랜덤하세 설정
b = tf.Variable(tf.random_uniform([1], -1, 1))
x_test = [5, 7]

# hypothesis 설정
hypothesis = w * x + b

# 오차를 계산하는 평균 제곱 오차 방법을 사용(경사 하강법에 사용하기 위해 계산)
cost = tf.reduce_mean(0.5 * (hypothesis - y) ** 2)  # cost function

# 옵티마이저 설정(학습 방법을 설정)
# 경사 하강법, w = w - a'E/'w     '는 편미분을 뜻함
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# 학습률이 작을경우 움직이는 거리가 작기 때문에 최적의 점으로 찾는데 오래걸림
# 학습률이 클경우 움직이는 거리가 크기 때문에 최적의 점을 찾지 못하고 넘어가서 발산할 수 있다.

# 학습을 시작
train = optimizer.minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# epoch를 10번 수행
for i in range(10000):
    sess.run(train)

print('w :', sess.run(w), 'b :', sess.run(b), 'cost :', sess.run(cost))

for i in x_test:
    print('x =', i, 'predict =', sess.run(w * i + b))

sess.close()
