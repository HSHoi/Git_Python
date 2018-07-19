import tensorflow as tf

x1 = [1., 0., 3., 0., 5.]  # 공부한 시간
x2 = [0., 2., 0., 4., 0.]  # 출석한 일수
y = [1., 2., 3., 4., 5.]  # 시험 점수

w1 = tf.Variable(tf.random_uniform([1], -1, 1))  # weight1
w2 = tf.Variable(tf.random_uniform([1], -1, 1))  # weight2
b = tf.Variable(tf.random_uniform([1], -1, 1))  # bias

# w1 = tf.Variable(0.)  # weight1
# w2 = tf.Variable(0.)  # weight2
# b = tf.Variable(0.)  # bias

hypothesis = w1 * x1 + w2 * x2 + b
cost = tf.reduce_mean(0.5 * (hypothesis - y) ** 2)  # Mean Squared Error
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)  # Gradient Descent
train = optimizer.minimize(loss=cost)  # train start

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('init_w1 :', sess.run(w1), 'init_w2 :', sess.run(w2), 'init_bias :', sess.run(b))

for i in range(1000):
    sess.run(train)
    print('epoch%5.d' % i, 'w1 :', sess.run(w1), 'w2 :', sess.run(w2), 'bias :', sess.run(b), 'cost :',
          sess.run(cost))
    if (sess.run(cost) < 0.001):  # early stop
        break

# print('Hypothesis [', sess.run(hypothesis), ']')
# print('w1 :', sess.run(w1), 'w2 :', sess.run(w2), 'b :', sess.run(b), 'cost :', sess.run(cost))

x_test = [0, 4]
W1, W2, bb = sess.run(w1), sess.run(w2), sess.run(b)
print('predict :', W1 * x_test[0] + W2 * x_test[1] + bb)
