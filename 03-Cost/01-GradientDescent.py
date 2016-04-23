import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 1 and b 0, but TensorFlow will figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))
b = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Out hypothesis
hypothesis = W * X

# Simplified cost function
# cost(W, b)
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
descent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(descent)

# Before starting, initialize the variables.
# we will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(100):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)
