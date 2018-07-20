import tensorflow as tf

x = [[1., 0., 3., 0., 5.], [0., 2., 0., 4., 0.]]  # 공부 시간, 출석 일수
y = [1., 2., 3., 4., 5.]  # 시험 점수
print(x)
# [[1.0, 0.0, 3.0, 0.0, 5.0], [0.0, 2.0, 0.0, 4.0, 0.0]]

x.insert(0, [1., 1., 1., 1., 1.])
print(x)
# [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 3.0, 0.0, 5.0], [0.0, 2.0, 0.0, 4.0, 0.0]]

w = tf.Variable(tf.random_uniform([1, 3], -1, 1))  # weight

hypothesis = tf.matmul(w, x)
cost = tf.reduce_mean(0.5 * (hypothesis - y) ** 2)  # Mean Squared Error
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)  # Gradient Descent
train = optimizer.minimize(loss=cost)  # train start

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('init_w :', sess.run(w))
# init_w : [[-0.47703314 -0.97613573 -0.8017659 ]]

for i in range(10000):
    sess.run(train)
    # print('cost :', sess.run(cost))

print('epoch%5.d' % i, 'w :', sess.run(w))
# epoch 9999 w : [[0.02080726 0.9945277  0.9935099 ]]

ww = sess.run(w)
x_test = [0, 4]
print(ww)
# [[0.02080726 0.9945277  0.9935099 ]]

print('predict :', x_test[0] * ww[0][1] + x_test[1] * ww[0][2] + ww[0][0])
# predict : 3.994846811518073
