import tensorflow as tf
import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

print('*'*30,'1')

mnist = input_data.read_data_sets('mnist', one_hot=True)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


drop_conv = tf.placeholder(tf.float32)
drop_hidden = tf.placeholder(tf.float32)        # 맨 마지막 적용
# 필터
# 3 x 3 x 1  convolution , output : 32
w1 = tf.Variable(tf.random_normal([3, 3,  1,  32], stddev=0.01))
# 3 x 3 x 32 convolution , output : 64
w2 = tf.Variable(tf.random_normal([3, 3, 32,  64], stddev=0.01))
# 3 x 3 x 64 convolution , output : 128
w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
# 128 X 4 X 4  input , output : 625
w4 = tf.Variable(tf.random_normal([128*4*4, 625], stddev=0.01))
# 625  x 10  output
w5 = tf.Variable(tf.random_normal([ 625,  10], stddev=0.01))

# (?, 28, 28, 32)    스트라이드는 배열 양 가장은 사용안함....
# 1칸, 1칸,  패딩은  원본의 크기를 줄이지 않겠다... 제로 패잉



# input : x,  filter : w,   스트라이드 : 1 x 1  -->   26 X 2 6 에 패딩 수행 --> 28  X 28

#input : [batch, in_height, in_width, in_channels] 형식. 28x28x1 형식의 손글씨 이미지.
#filter : [filter_height, filter_width, in_channels, out_channels] 형식. 3, 3, 1, 32의 w.
#strides : 크기 4인 1차원 리스트. [0], [3]은 반드시 1. 일반적으로 [1], [2]는 같은 값 사용.
#padding : 'SAME' 또는 'VALID'. 패딩을 추가하는 공식의 차이. SAME은 출력 크기를 입력과 같게 유지
# 3x3x1 필터를 32개 만드는 것


# z1 : 128X 28 X 28 X 32 : (입력 크기 X 필터 채널 크기)
# z1(128 X 28 X 28 X 32)  w1(3 X 3 X  1 X 32)
z1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
# r1 : 128X 28 X 28 X 32 :
r1 = tf.nn.relu(z1)


# (?, 14, 14, 32)    28 X 28 --> 2 X 2 맥스 풀링 수행 : 14 x 14
#value : [batch, height, width, channels] 형식의 입력 데이터. ReLU를 통과한 출력 결과가 된다.
#ksize : 4개 이상의 크기를 갖는 리스트로 입력 데이터의 각 차원의 윈도우 크기.
#data_format : NHWC 또는 NCHW. n-count, height, width, channel의 약자 사용.
#ksize가 [1,2,2,1]이라는 뜻은 2칸씩 이동하면서 출력 결과를 1개 만들어 낸다는 것이다. 다시 말해 4개의 데이터 중에서 가장 큰 1개를 반환하는 역할

# P1 : 128 X 14 X 14 X32
p1 = tf.nn.max_pool(r1, ksize=[1, 2, 2, 1],     # ksize : 필터의 크기 (2x2)   값은 없음..
                    strides=[1, 2, 2, 1], padding='SAME')
# d1 : 128 X 14 X 14 X32
d1 = tf.nn.dropout(p1, drop_conv)

# (?, 14, 14, 64)         슬트이드 : 1 x 1  --> 12 x 12 에 패딩 수행 : 14 X 14
#    d1 (128 X 14 X 14 X 32 )  w2(3 X 3X 32 X  64)
#  z2 (128 X 14 X 14 X 64)
z2 = tf.nn.conv2d(d1, w2, strides=[1, 1, 1, 1], padding='SAME')
# r2 (128 X 14X 14 X 64)
r2 = tf.nn.relu(z2)

# P2 (128, 7, 7, 64)    : 14 X 14 --> 2 X 2 맥스풀링  수행 : 7 X 7
p2 = tf.nn.max_pool(r2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
# d2 (128, 7, 7, 64)
d2 = tf.nn.dropout(p2, drop_conv)


# d2(128, 7, 7, 128)  w3(3 X 3X 64X 128)
# 슬라이드 : 1 x 1    --> 5 X 5에 패딩 수행 : 7 X 7
# z3(128 X 7 X 7 X 128)
z3 = tf.nn.conv2d(d2, w3, strides=[1, 1, 1, 1], padding='SAME')
# r3 (128 X 7 X 7 X 128)
r3 = tf.nn.relu(z3)

#
# p3(128, 4, 4, 128)          7 X 7 --> 2 X 2 맥스 풀링 수행    : 4 X 4
p3 = tf.nn.max_pool(r3, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
# d3(128 X 4 X 4 X 128)
d3 = tf.nn.dropout(p3, drop_conv)

# 2차원 --> 1차원
#   4 X 4 X 128 -->  2048
#d3(128 X 2048)
d3 = tf.reshape(d3, [-1, 2048])
#  (128 x  2048 )  * (2048 X 625)  -->  128 X 625
# z4, r4, d4 (128 X 625)  w4(2048 X 625)
z4 = tf.matmul(d3, w4)
r4 = tf.nn.relu(z4)
d4 = tf.nn.dropout(r4, drop_hidden)

# 출력층에 대한 hypothesis
# ( 128 X 625)       * (625 X 10)  -->  128 X 10
hypothesis = tf.matmul(d4, w5)
# y = 128 X 10
# softmax를 통해 크로스 엔트로피를 구한다.
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis,
                                                 labels=y)
cost = tf.reduce_mean(cost_i)

#optimizer = tf.train.RMSPropOptimizer(0.001)
optimizer = tf.train.AdadeltaOptimizer (0.001)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


#print("w1 :", sess.run(w1))
print('*'*30,'2')

epochs, batch_size = 200, 128         # 200, 128
for i in range(epochs):
    total = 0
    count = mnist.train.num_examples // batch_size # batch size 를 사용하기 때문에  epoch 를 변경해야함
    print("count :", count, ' mnist.train.num_examples :', mnist.train.num_examples)

    print('*' * 30, '3')
    for j in range(count):
        # 배치 싸이즈 만큼씩 읽어온다.
        xx, yy = mnist.train.next_batch(batch_size)
        # xx shape : 128 X 784  (배치 사이즈 * 이미지 사이즈)
        # yy shape : 128 X 10 (one - hot encoding )

        #print( 'xx.shape :', xx.shape , ' xx :' , xx)
        #print( 'yy.shape :', yy.shape , ' yy :' , yy)
        # 128 * 28 * 28 (배치 사이즈 X 이미지 사이즈(28 X 28)
        xx = xx.reshape(-1, 28, 28, 1)        # 4차원 변환  -1 (모른다는 의미)  28 x 28 로 차원 증가.


        #print( '22 xx.shape :', xx.shape )
        c, _ = sess.run([cost, train],
                        feed_dict={x: xx,
                                   y: yy,
                                   drop_conv: 0.8,
                                   drop_hidden: 1.0})
        total += c

        #print("xx shape :", xx.shape)

        #print('#' * 30, '1')
        #print('z1.shape :', sess.run(z1,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'r1.shape :', sess.run(r1,feed_dict={x: xx,drop_conv: 0.8}).shape)
        #print('p1.shape :', sess.run(p1,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'd1.shape :', sess.run(d1,feed_dict={x: xx,drop_conv: 0.8}).shape)
        #print('z2.shape :', sess.run(z2,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'r2.shape :', sess.run(r2,feed_dict={x: xx,drop_conv: 0.8}).shape)
        #
        #print('p2.shape :', sess.run(p2,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'd2.shape :', sess.run(d2,feed_dict={x: xx,drop_conv: 0.8}).shape)
        #print('z3.shape :', sess.run(z3,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'r3.shape :', sess.run(r3,feed_dict={x: xx,drop_conv: 0.8}).shape)
        #print('p3.shape :', sess.run(p3,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'd3.shape :', sess.run(d3,feed_dict={x: xx,drop_conv: 0.8}).shape)
        #print('z4.shape :', sess.run(z4,feed_dict={x: xx,drop_conv: 0.8}).shape,
        #      'r4.shape :', sess.run(r4,feed_dict={x: xx,drop_conv: 0.8}).shape)
        #print('d4.shape :', sess.run(d4,feed_dict={x: xx,drop_conv: 0.8,drop_hidden: 1.0}).shape)

        #print('#' * 30, '1')

#        if j % 10 == 0:
#            print('{} / {}'.format(j, count))

    print('{:3} : {}'.format(i+1, total / count))

    # ----------------------- #

    total_indices = np.arange(mnist.test.num_examples)
    np.random.shuffle(total_indices)

    test_size = 256
    test_indices = total_indices[:test_size]    # 섞은 데이터 중 256개 배열을 가져옴...

    test_xx = mnist.test.images[test_indices].reshape(-1, 28, 28, 1) # 256 장의 데이터 가져와  4차원 변환

    label = np.argmax(mnist.test.labels[test_indices], axis=1)
    y_hat = sess.run(tf.argmax(hypothesis, axis=1),
                     feed_dict={x: test_xx,
                                drop_conv: 1.0,
                                drop_hidden: 1.0})

    print('{:3} : {:.7f}'.format(i+1, np.mean(label == y_hat)))

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
