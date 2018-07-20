import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

# one hot Encoding 으로 데이터 읽기
mnist = input_data.read_data_sets('mnist', one_hot=True)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

drop_conv = tf.placeholder(tf.float32)
drop_hidden = tf.placeholder(tf.float32)

# 필터 선언
w1 = tf.Variable(tf.random_uniform([3, 3, 1, 32], stddev=0.01))
w2 = tf.Variable(tf.random_uniform([3, 3, 32, 64], stddev=0.01))
w3 = tf.Variable(tf.random_uniform([3, 3, 64, 128], stddev=0.01))

# 가중치 선언
w4 = tf.Variable(tf.random_uniform([128 * 4 * 4, 625], stddev=0.01))

# 출력층 적용을 위한 가중치
w5 = tf.Variable(tf.random_uniform([625, 10], stddev=0.01))

z1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')  # 입력값과 크기가 똑같이 padding 설정
r1 = tf.nn.relu(z1)
p1 = tf.nn.max_pool(r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
d1 = tf.nn.dropout(p1, drop_conv)

print('*' * 30, '1')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# print("w1 :", sess.run(w1))
print('*' * 30, '2')

epochs, batch_size = 2, 128  # 200, 128
for i in range(epochs):
    total = 0
    count = mnist.train.num_examples // batch_size
    print("count :", count, ' mnist.train.num_examples :', mnist.train.num_examples)

    print('*' * 30, '3')
    for j in range(count):
        # 배치 싸이즈 만큼씩 읽어온다.
        xx, yy = mnist.train.next_batch(batch_size)
        # xx shape : 128 X 784  (배치 사이즈 * 이미지 사이즈)
        # yy shape : 128 X 10 (one - hot encoding )

        # print( 'xx.shape :', xx.shape , ' xx :' , xx)
        # print( 'yy.shape :', yy.shape , ' yy :' , yy)
        # 128 * 28 * 28 (배치 사이즈 X 이미지 사이즈(28 X 28)
        xx = xx.reshape(-1, 28, 28, 1)  # 4차원 변환  -1 (모른다는 의미)  28 x 28 로 차원 증가.

        # print( '22 xx.shape :', xx.shape )
        c, _ = sess.run([cost, train],
                        feed_dict={x: xx,
                                   y: yy,
                                   drop_conv: 0.8,
                                   drop_hidden: 1.0})
        total += c

        # print("xx shape :", xx.shape)

        # print('#' * 30, '1')
        # print('z1.shape :', sess.run(z1,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'r1.shape :', sess.run(r1,feed_dict={x: xx,drop_conv: 0.8}).shape)
        # print('p1.shape :', sess.run(p1,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'd1.shape :', sess.run(d1,feed_dict={x: xx,drop_conv: 0.8}).shape)
        # print('z2.shape :', sess.run(z2,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'r2.shape :', sess.run(r2,feed_dict={x: xx,drop_conv: 0.8}).shape)
        #
        # print('p2.shape :', sess.run(p2,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'd2.shape :', sess.run(d2,feed_dict={x: xx,drop_conv: 0.8}).shape)
        # print('z3.shape :', sess.run(z3,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'r3.shape :', sess.run(r3,feed_dict={x: xx,drop_conv: 0.8}).shape)
        # print('p3.shape :', sess.run(p3,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'd3.shape :', sess.run(d3,feed_dict={x: xx,drop_conv: 0.8}).shape)
        # print('z4.shape :', sess.run(z4,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'r4.shape :', sess.run(r4,feed_dict={x: xx,drop_conv: 0.8}).shape)
        # print('d4.shape :', sess.run(d4,feed_dict={x: xx,drop_conv: 0.8,drop_hidden: 1.0}).shape)

        # print('#' * 30, '1')

    #        if j % 10 == 0:
    #            print('{} / {}'.format(j, count))

    print('{:3} : {}'.format(i + 1, total / count))

    # ----------------------- #

    total_indices = np.arange(mnist.test.num_examples)
    np.random.shuffle(total_indices)

    test_size = 256
    test_indices = total_indices[:test_size]  # 섞은 데이터 중 256개 배열을 가져옴...

    test_xx = mnist.test.images[test_indices].reshape(-1, 28, 28, 1)  # 256 장의 데이터 가져와  4차원 변환

    label = np.argmax(mnist.test.labels[test_indices], axis=1)
    y_hat = sess.run(tf.argmax(hypothesis, axis=1),
                     feed_dict={x: test_xx,
                                drop_conv: 1.0,
                                drop_hidden: 1.0})

    print('{:3} : {:.7f}'.format(i + 1, np.mean(label == y_hat)))

# -------------------------- #

# 10000 X 28 X 28 X 1
xx = mnist.test.images.reshape(-1, 28, 28, 1)

print(" 정확도 확인을 위한 xx :", xx.shape)
y_hat = sess.run(hypothesis, feed_dict={x: xx,
                                        drop_conv: 1.0,
                                        drop_hidden: 1.0})
# 행에 대해 최대값 구해오기 [ 0, 0, 0 , 1 ] --> 3
y_hat_arg = np.argmax(y_hat, axis=1)
y_arg = np.argmax(mnist.test.labels, axis=1)
print('y_arg shape :', y_arg.shape)
print('y_arg :', y_arg)
print('y_hat_arg :', y_hat_arg)
print('mnist.test.labels :', mnist.test.labels)

print('accuracy :', np.mean(y_hat_arg == y_arg))
sess.close()
