import tensorflow as tf

FEATURE_SIZE = 64

class fSIM:
	def __init__(self, batch_size):
		self.batch_size = batch_size
		with tf.name_scope('inputs'):
			self.in_x = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 64, 1], name='in_x')
			self.in_y = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 64, 3], name='in_y')
			self.in_z = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 64, 3], name='in_z')

		with tf.name_scope("RetrievalNet"):
			self.out_x1, self.out_x2, self.out_x3 = self.RetrievalNet(self.in_x)

		with tf.variable_scope("EmbeddingNet", reuse=tf.AUTO_REUSE):
			self.out_y = self.EmbeddingNet(self.in_y)
			self.out_z = self.EmbeddingNet(self.in_z)

		with tf.name_scope('Loss'):
			self.loss = self.compute_loss()
			tf.summary.scalar('Loss', self.loss)
		print("FuncNet initialized")

	def RetrievalNet(self, x):
		conv_x0 = self.conv3d_layer(x, 32, 'conv_x0')
		conv_x0_out = tf.nn.relu(conv_x0)
		conv_x1 = self.conv3d_layer(conv_x0_out, 64, 'conv_x1')
		conv_x1_out = tf.nn.relu(conv_x1)
		conv_x2 = self.conv3d_layer(conv_x1_out, 128, 'conv_x2')
		conv_x2_out = tf.nn.relu(conv_x2)
		conv_x3 = self.conv3d_layer(conv_x2_out, 256, 'conv_x3')
		conv_x3_out = tf.nn.relu(conv_x3)

		fc_x_in = tf.reshape(conv_x3_out, [self.batch_size, -1])
		fc_x0 = self.fc_layer(fc_x_in, 3, "fc_x0")
		fc_x0_out = tf.nn.softmax(fc_x0)
		fc_x1 = self.fc_layer(fc_x_in, 3*FEATURE_SIZE, "fc_x1")
		fc_x1_out = fc_x1
		fc_x2 = self.fc_layer(fc_x_in, 3*FEATURE_SIZE, "fc_x2")
		fc_x2_out = tf.minimum(tf.exp(fc_x2), 0.365)
		return fc_x0_out, fc_x1_out, fc_x2_out

	def EmbeddingNet(self, y):
		conv_y0 = self.conv3d_layer(y, 16, 'conv_y0')
		conv_y0_out = tf.nn.relu(conv_y0)
		conv_y1 = self.conv3d_layer(conv_y0_out, 32, 'conv_y1')
		conv_y1_out = tf.nn.relu(conv_y1)
		conv_y2 = self.conv3d_layer(conv_y1_out, 64, 'conv_y2')
		conv_y2_out = tf.nn.relu(conv_y2)
		conv_y3 = self.conv3d_layer(conv_y2_out, 128, 'conv_y3')
		conv_y3_out = tf.nn.relu(conv_y3)

		fc_y_in = tf.reshape(conv_y3_out, [self.batch_size, -1])
		fc_y = self.fc_layer(fc_y_in, FEATURE_SIZE, "fc_y")
		return fc_y

	def conv3d_layer(self, inputs, out_dim, name, k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2):
		with tf.name_scope('conv_layer'):
			with tf.name_scope('weights'):
				weights = tf.get_variable(name=name + '/weights', shape=[k_d, k_h, k_w, inputs.get_shape()[-1], out_dim], initializer=tf.contrib.layers.xavier_initializer())
				tf.summary.histogram(name + '/weights', weights)
			with tf.name_scope('biases'):
				biases = tf.get_variable(name=name + '/biases', shape=[out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
				tf.summary.histogram(name + '/biases', biases)
			with tf.name_scope('conv_out'):
				conv = tf.nn.bias_add(tf.nn.conv3d(inputs, weights, strides=[1, d_d, d_h, d_w, 1], padding='SAME'), biases)
		return conv

	def fc_layer(self, inputs, out_dim, name):
		assert len(inputs.get_shape()) == 2
		with tf.name_scope('fc_layer'):
			with tf.name_scope('weights'):
				weights = tf.get_variable(name=name + '/weights', dtype=tf.float32, shape=[inputs.get_shape()[1], out_dim], initializer=tf.contrib.layers.xavier_initializer())
				tf.summary.histogram(name + '/weights', weights)
			with tf.name_scope('biases'):
				biases = tf.get_variable(name=name + '/biases', shape=[out_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
				tf.summary.histogram(name + '/biases', biases)
			with tf.name_scope('fc_out'):
				fc = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
		return fc

	def compute_loss(self):
		w1, w2, w3 = tf.split(self.out_x1, 3, 1)
		self.weight1 = w1 / (w1 + w2 + w3)
		self.weight2 = w2 / (w1 + w2 + w3)
		self.weight3 = w3 / (w1 + w2 + w3)
		self.mu1, self.mu2, self.mu3 = tf.split(self.out_x2, 3, 1)
		self.stddev1, self.stddev2, self.stddev3 = tf.split(self.out_x3, 3, 1)
		mvn1 = tf.contrib.distributions.MultivariateNormalDiag(self.mu1, self.stddev1, allow_nan_stats=False)
		mvn2 = tf.contrib.distributions.MultivariateNormalDiag(self.mu2, self.stddev2, allow_nan_stats=False)
		mvn3 = tf.contrib.distributions.MultivariateNormalDiag(self.mu3, self.stddev3, allow_nan_stats=False)
		e1 = - tf.log(mvn1.prob(self.out_y) * self.weight1 +
						mvn2.prob(self.out_y) * self.weight2 +
						mvn3.prob(self.out_y) * self.weight3 + 1.0e-10)
		e2 = - tf.log(mvn1.prob(self.out_z) * self.weight1 +
						mvn2.prob(self.out_z) * self.weight2 +
						mvn3.prob(self.out_z) * self.weight3 + 1.0e-10)

		loss = tf.reduce_mean(tf.maximum(10 + e1 - e2, 0), name="loss")
		return loss
