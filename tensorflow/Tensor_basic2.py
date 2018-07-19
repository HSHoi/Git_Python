import tensorflow as tf

# graph field
aa = tf.placeholder(tf.float32)  # tf.placeholder 은 변수 형태만 설정해줌(int, float 같은 형들)
b = tf.placeholder(tf.float32)  # 변수 형을 알려줌
add = tf.add(aa, b)  # 다른 데이터 형이 들어올 경우 오류 뱉음

# start field
sess = tf.Session()
sess.run(tf.global_variables_initializer())  # graph field 에 있는 변수들을 초기화(무조건 씀)
print(sess.run(add, feed_dict={aa: 4, b: 5}))
# 9.0

sess.close()
