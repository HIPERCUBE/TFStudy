import tensorflow as tf

# Create hello world using TensorFlow
# The op is added as node to the default graph.
#
# The value returned by the constructor represents the ouput
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

print hello

# Start tf session
sess = tf.Session()

print sess.run(hello)