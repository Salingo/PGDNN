import tensorflow as tf

DIM = 64

class iSEG:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.obj_cate_num = 20 # central1 + inter num18 + free1

        with tf.name_scope('inputs'):
            self.x = tf.placeholder(
                tf.float32, [self.batch_size, DIM, DIM, DIM, 3], name='x_in')
            self.x_io = tf.placeholder(
                tf.float32, [self.batch_size, DIM, DIM, DIM, self.obj_cate_num], name='x_io_in') # ground truth
            self.x_binary = tf.placeholder(
                tf.float32, [self.batch_size, DIM, DIM, DIM, 2], name='x_binary_cf')
            self.label = tf.placeholder(
                tf.float32, [self.batch_size, 25], name='label')
            self.label_idx = tf.placeholder(
                tf.float32, [self.batch_size, self.obj_cate_num], name='label_idx')

        with tf.name_scope('encoder'):
            self.z, out_e4, out_e3, out_e2, out_e1 = self.encoder(self.x, self.label)

        with tf.name_scope('decoder'):
            self.x_rec = self.decoder(self.z, out_e4, out_e3, out_e2, out_e1)
        
        with tf.name_scope('loss'):
            # concat central obj and free space
            self.x_rec = tf.concat([tf.expand_dims(self.x_binary[:,:,:,:,0],-1),self.x_rec,tf.expand_dims(self.x_binary[:,:,:,:,-1],-1)],-1)
            self.x_rec = tf.nn.softmax(self.x_rec,dim=-1)

            crosse_loss = -tf.reduce_mean(self.x_io * tf.log(self.x_rec + 1e-9))
            self.loss = tf.reduce_mean(crosse_loss)

            # tf.summary.scalar('Loss', self.loss)

        print("Classifier initialized")

    def encoder(self, x, label):
        # encode x
        conv_e1 = self.conv3d_layer(x, 32, 'conv_e1')
        out_e1 = tf.nn.relu(conv_e1)
        conv_e2 = self.conv3d_layer(out_e1, 64, 'conv_e2')
        out_e2 = tf.nn.relu(conv_e2)
        conv_e3 = self.conv3d_layer(out_e2, 128, 'conv_e3')
        out_e3 = tf.nn.relu(conv_e3)
        conv_e4 = self.conv3d_layer(out_e3, 256, 'conv_e4')
        out_e4 = tf.nn.relu(conv_e4)
        fc_e5 = tf.reshape(out_e4, [self.batch_size, -1])
        fc_e5 = self.fc_layer(fc_e5, 128, 'fc_e5')
        out_fc_e5 = tf.nn.relu(fc_e5)
        fx = out_fc_e5

        # encode c
        fc_e6 = self.fc_layer(label, 128, 'fc_e6')
        out_fc_e6 = tf.nn.relu(fc_e6)
        fc_e7 = self.fc_layer(out_fc_e6, 128, 'fc_e7')
        out_fc_e7 = tf.nn.relu(fc_e7)
        fc_e8 = self.fc_layer(out_fc_e7, 128, 'fc_e8')
        out_fc_e8 = tf.nn.relu(fc_e8)
        fc = out_fc_e8

        # concat fx, fc
        z = tf.concat([fx, fc],-1)
        z = self.fc_layer(z, 256, 'fc_e9')
        z = tf.nn.relu(z)
        z = self.fc_layer(z, 256, 'fc_e10')
        z = tf.nn.relu(z)
        print(z)

        return z, out_e4, out_e3, out_e2, out_e1

    def decoder(self, z, out_e4, out_e3, out_e2, out_e1):
        # d1 = self.fc_layer(z, 256*4*4*4, 'fc_d')
        d1 = tf.reshape(z, (self.batch_size, 1, 1, 1, 256))
        d1 = self.conv3d_trans_layer(d1, 256, (self.batch_size, 4, 4, 4, 256), 'conv_trans_d1', d_h=1, d_w=1, d_d=1, padding="VALID")
        out_d1 = tf.nn.relu(d1)

        out_d1 = tf.concat([out_d1,out_e4], 4)
        conv_d2 = self.conv3d_trans_layer(out_d1, 128, (self.batch_size, 8, 8, 8, 128), 'conv_trans_d2')
        out_d2 = tf.nn.relu(conv_d2)

        out_d2 = tf.concat([out_d2,out_e3], 4)
        conv_d3 = self.conv3d_trans_layer(out_d2, 64, (self.batch_size, 16, 16, 16, 64), 'conv_trans_d3')
        out_d3 = tf.nn.relu(conv_d3)

        out_d3 = tf.concat([out_d3,out_e2], 4)
        conv_d4 = self.conv3d_trans_layer(out_d3, 32, (self.batch_size, 32, 32, 32, 32), 'conv_trans_d4')
        out_d4 = tf.nn.relu(conv_d4)
 
        out_d4 = tf.concat([out_d4,out_e1], 4)
        conv_d5 = self.conv3d_trans_layer(out_d4, self.obj_cate_num-2, (self.batch_size, DIM, DIM, DIM, self.obj_cate_num-2), 'conv_trans_d5')
        out_d5 = tf.nn.softmax(conv_d5,dim=-1)
 
        return out_d5

    def conv3d_trans_layer(self, inputs, out_dim, output_shape, name, k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2, padding='SAME'):
        with tf.name_scope('conv_trans_layer'):
            with tf.name_scope('weights'):
                weights = tf.get_variable(name=name + '/weights', 
                        shape=[k_d, k_h, k_w, out_dim, inputs.get_shape()[-1]], initializer=tf.contrib.layers.xavier_initializer())
                tf.summary.histogram(name + '/weights', weights)
            with tf.name_scope('conv_trans_out'):
                conv_trans = tf.nn.conv3d_transpose(inputs, weights, output_shape, strides=[1, d_d, d_h, d_w, 1], padding=padding)
        return conv_trans

    def conv3d_layer(self, inputs, out_dim, name, k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2, padding='SAME'):
        with tf.name_scope('conv_layer'):
            with tf.name_scope('weights'):
                weights = tf.get_variable(name=name + '/weights', 
                        shape=[k_d, k_h, k_w, inputs.get_shape()[-1], out_dim], initializer=tf.contrib.layers.xavier_initializer())
                tf.summary.histogram(name + '/weights', weights)
            with tf.name_scope('conv_out'):
                conv = tf.nn.conv3d(inputs, weights, strides=[1, d_d, d_h, d_w, 1], padding=padding)
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