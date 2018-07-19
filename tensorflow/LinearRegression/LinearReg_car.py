import tensorflow as tf
import numpy as np

cars = np.loadtxt('cars.csv', delimiter=',', unpack=True)
x = cars[0]
y = cars[0]

w = tf.Variable(tf.random_uniform([1], -1, 1))  # weight
b = tf.Variable(tf.random_uniform([1], -1, 1))  # bias

hypothesis = w * x + b  # hypothesis model
cost = tf.reduce_mean(0.5 * (hypothesis - y) ** 2)  # cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)  # gradientdescent method
train = optimizer.minimize(loss=cost)  # train start

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train)

ww, bb = sess.run(w), sess.run(b)
print('w :', ww, 'b :', bb, 'cost :', sess.run(cost))
# w : [0.9867082] b : [0.2282526] cost : 0.002697913

sess.close()
