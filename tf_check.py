import tensorflow as tf


input = tf.Variable(tf.ones([1, 29, 29, 1]))
filter = tf.Variable(tf.ones([7, 7, 1, 1]))
op = tf.nn.conv2d(input, filter, strides = [1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	rs = (sess.run(op))
	print('res.shape')
	print(res.shape)
	output = tf.reshape(res, [6, 6])
	print(sess.run(output))