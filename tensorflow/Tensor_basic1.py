import tensorflow as tf

a = tf.constant(3)  # a 를 상수형 3 으로 설정해줌, 초기화를 하지 않아도 됨
b = tf.Variable(5)  # b 를 변수형으로 5를 설정, 초기화를 해야함(tf.global_variables_initializer())

add = tf.add(a, b)

print('a : ', a, 'b :', b)
# a :  Tensor("Const:0", shape=(), dtype=int32) b : <tf.Variable 'Variable:0' shape=() dtype=int32_ref>

sess = tf.Session()

# 초기화(전체 변수 초기화)
sess.run(tf.global_variables_initializer())
print('a :', sess.run(a))
# a : 3

print('b :', sess.run(b))
# b : 5

print('add :', sess.run(add))
# add : 8

sess.close()
