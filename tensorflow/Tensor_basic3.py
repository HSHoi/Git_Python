import tensorflow as tf

value = tf.Variable(0)
one = tf.constant(1)
state = tf.add(value, one)

# state 값을 value 에 할당하고 할당된 값을 리턴하기 위한 함수 assign
update = tf.assign(value, state)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for _ in range(3):
    # print('update :', sess.run(update), 'state :', sess.run(state))
    ## update : 1 state : 2
    ## update : 2 state : 3
    ## update : 3 state : 4

    print('update :', sess.run(state), 'state :', sess.run(update))
    # update : 1 state : 1
    # update : 2 state : 2
    # update : 3 state : 3
