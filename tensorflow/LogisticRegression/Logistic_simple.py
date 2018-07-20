import tensorflow as tf
import numpy as np

# x = 6 x 3
x = [[1., 1., 2.],
     [1., 2., 3.],
     [1., 2., 4.],
     [1., 5., 3.],
     [1., 7., 5.],
     [1., 8., 4.]]

# y = 6 x 1
y = [[0], [0], [0], [1], [1], [1]]

# w = 3 x 1
w = tf.Variable(tf.random_uniform([3, 1], -1, 1))
yy = np.array(y)

# z = 6 x 3 * 3 x 1 = 6 x 1
z = tf.matmul(x, w)

hypothesis = 1 / (1 + tf.exp(-z))  # Logistic Regression
# Logistic Regression loss function
cost = tf.reduce_mean(yy - tf.log(hypothesis) + (1 - yy) * (-tf.log(1 - hypothesis)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss=cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train)
    print('epoch :', i, 'cost :', sess.run(cost))

sess.close()
