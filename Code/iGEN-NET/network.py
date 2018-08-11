import tensorflow as tf
import numpy as np
from transformer import spatial_transformer_network as stn

# weights for 25 categories
WEIGHT = ['0.528-0.3724-0.0996','0.4405-0.4551-0.1044','0.3818-0.4807-0.1375','0.3293-0.5233-0.1475','0.4463-0.4274-0.1262','0.5692-0.3382-0.0927','0.4096-0.4234-0.1671',
          '0.4938-0.3693-0.1369','0.5246-0.3724-0.103','0.5149-0.366-0.119','0.5712-0.3328-0.0961','0.5885-0.3115-0.1','0.6988-0.2378-0.0634','0.7601-0.1939-0.0461',
          '0.5339-0.3565-0.1096','0.6251-0.258-0.1169','0.5316-0.3588-0.1096','0.3998-0.4875-0.1127','0.3661-0.4952-0.1386','0.5867-0.3272-0.0861','0.5706-0.3296-0.0998',
          '0.5507-0.3404-0.1089','0.5649-0.3274-0.1077','0.5917-0.313-0.0953','0.5151-0.3782-0.1067']
DIM = 64
LEAKY_VALUE = 0.2
CATE_DIM = 25

class iGEN:
    def __init__(self, batch_size, bn_set):
        self.batch_size = batch_size
        self.is_training = bn_set

        with tf.name_scope('inputs'):
            self.x_sce = tf.placeholder(
                tf.float32, [self.batch_size, DIM, DIM, DIM, 3], name='x_sce')
            self.x_obj = tf.placeholder(
                tf.float32, [self.batch_size, DIM, DIM, DIM, 1], name='x_obj')
            self.label = tf.placeholder(
                tf.float32, [self.batch_size, CATE_DIM], name = 'label')
            self.S = tf.placeholder(tf.float32, [self.batch_size], name = 'S')
            self.T = tf.placeholder(tf.float32, [self.batch_size, 3], name = 'T')

        with tf.variable_scope('encoder', reuse = tf.AUTO_REUSE):
            z = self.encoder(self.x_obj, self.label, is_training=self.is_training)
        
        with tf.variable_scope('decoder'):
            self.iobj = self.decoder_iobj(z, is_training=self.is_training)
            self.w_scale, self.w_trans = self.decoder_w(z)

        with tf.variable_scope('scale'):
            self.cobj = self.scale(self.x_obj, self.w_scale, self.w_trans)

        cobj = tf.minimum(self.cobj*100, 1)
        x_rec = tf.concat([cobj, self.iobj], -1)
        self.x_rec = tf.nn.softmax(x_rec, -1)

        with tf.name_scope('loss'):
            x_sce_list = tf.unstack(self.x_sce,axis=0)
            x_rec_list = tf.unstack(self.x_rec,axis=0)
            label_list = tf.unstack(self.label,axis=0)
            rec_loss_list = []
            for i in range(self.batch_size):
                temp = WEIGHT[np.argmax(label_list[i])].split('-')
                w = []
                for b in temp:
                    w.append(float(b))
                rec_w = tf.constant(w,dtype=tf.float32)
                rec_loss = -tf.reduce_mean(rec_w*(x_sce_list[i] * tf.log(x_rec_list[i] + 1e-9)))
                rec_loss_list.append(rec_loss)
            rec_loss = tf.stack(rec_loss_list,axis=0)

            self.rec_loss = tf.reduce_mean(rec_loss)

            ts_loss = self.dis(self.S, self.w_scale) + self.dis(self.T, self.w_trans)
            self.ts_loss = tf.reduce_mean(ts_loss)

            self.loss = self.ts_loss + self.rec_loss

            tf.summary.scalar('ts_loss', self.ts_loss)
            tf.summary.scalar('rec_loss', self.rec_loss)
            tf.summary.scalar('loss', self.loss)

        print('batch_size:',batch_size,'bn',bn_set)
    
    def encoder(self, x, label, is_training=True):
        # encode x
        conv_e1 = self.conv3d_layer(x, 32, 'conv_e1')
        out_e1 = tf.nn.relu(conv_e1)
        conv_e2 = self.conv3d_layer(out_e1, 64, 'conv_e2')
        out_e2 = tf.nn.relu(self.batch_norm(conv_e2, is_training, 'bn_e2'))
        conv_e3 = self.conv3d_layer(out_e2, 128, 'conv_e3')
        out_e3 = tf.nn.relu(self.batch_norm(conv_e3, is_training, 'bn_e3'))
        conv_e4 = self.conv3d_layer(out_e3, 256, 'conv_e4')
        out_e4 = tf.nn.relu(self.batch_norm(conv_e4, is_training, 'bn_e4'))

        fc_e5 = tf.reshape(out_e4, [self.batch_size, -1])
        fc_e5 = self.fc_layer(fc_e5, 128, 'fc_e5')
        out_fc_e5 = tf.nn.relu(self.batch_norm(fc_e5, is_training, 'bn_e5'))
        fx = out_fc_e5

        # encode c
        fc_e6 = self.fc_layer(label, 128, 'fc_e6')
        out_fc_e6 = tf.nn.relu(self.batch_norm(fc_e6, is_training, 'bn_e6'))
        fc_e7 = self.fc_layer(out_fc_e6, 128, 'fc_e7')
        out_fc_e7 = tf.nn.relu(self.batch_norm(fc_e7, is_training, 'bn_e7'))
        fc_e8 = self.fc_layer(out_fc_e7, 128, 'fc_e8')
        out_fc_e8 = tf.nn.relu(self.batch_norm(fc_e8, is_training, 'bn_e8'))
        fc = out_fc_e8

        # concat fx, fc
        z = tf.concat([fx, fc],-1)
        z = self.fc_layer(z, 256, 'fc_e9')
        z = tf.nn.relu(self.batch_norm(z, is_training, 'bn_e9'))
        z = self.fc_layer(z, 256, 'fc_e10')
        z = tf.nn.relu(self.batch_norm(z, is_training, 'bn_e10'))

        return z

    def decoder_iobj(self, z, is_training=True):
        d1 = tf.reshape(z, (self.batch_size, 1, 1, 1, 256))
        d1 = self.conv3d_trans_layer(d1, 256, (self.batch_size, 4, 4, 4, 256), 'conv_trans_d1', d_h=1, d_w=1, d_d=1, padding='VALID')
        out_d1 = tf.nn.relu(d1)
        conv_d2 = self.conv3d_trans_layer(out_d1, 128, (self.batch_size, 8, 8, 8, 128), 'conv_trans_d2')
        out_d2 = tf.nn.relu(self.batch_norm(conv_d2, is_training, 'bn_d2'))
        conv_d3 = self.conv3d_trans_layer(out_d2, 64, (self.batch_size, 16, 16, 16, 64), 'conv_trans_d3')
        out_d3 = tf.nn.relu(self.batch_norm(conv_d3, is_training, 'bn_d3'))
        conv_d4 = self.conv3d_trans_layer(out_d3, 32, (self.batch_size, 32, 32, 32, 32), 'conv_trans_d4')
        out_d4 = tf.nn.relu(self.batch_norm(conv_d4, is_training, 'bn_d4'))
        conv_d5 = self.conv3d_trans_layer(out_d4, 2, (self.batch_size, DIM, DIM, DIM, 2), 'conv_trans_d5')
        out_d5 = tf.nn.sigmoid(conv_d5)

        return out_d5

    def decoder_w(self, z):
        fc_w1 = self.fc_layer(z, 64, 'fc_w1')
        fc_w1 = tf.nn.relu(fc_w1)
        fc_w2 = self.fc_layer(fc_w1, 64, 'fc_w2')
        fc_w2 = tf.nn.relu(fc_w2)
        fc_w3 = self.fc_layer(fc_w2, 4, 'fc_w3')
        scale = fc_w3[:,0]
        translate = fc_w3[:,1:]

        return scale, translate

    def scale(self, x_obj, w_scale, w_trans):
        # removes the last dimension
        obj = tf.squeeze(x_obj)
        w_scale = tf.expand_dims(w_scale, -1)
        w_trans_x = tf.expand_dims(w_trans[:,0], -1)
        w_trans_y = tf.expand_dims(w_trans[:,1], -1)
        w_trans_z = tf.expand_dims(w_trans[:,2], -1)

        # !!! note that in their implementation for images, y and x are swiiched, 
        # in out implementation it as well: [batch, y, x, z]

        # transform the x and y dim first
        s_xy = tf.matmul(w_scale, tf.constant([[1,0,0,0,1,0]],dtype=tf.float32))
        t_xy = tf.matmul(w_trans_x, tf.constant([[0,0,1,0,0,0]],dtype=tf.float32)) + tf.matmul(w_trans_y, tf.constant([[0,0,0,0,0,1]],dtype=tf.float32))
        T_xy = s_xy + t_xy
        transformed_xy = stn(obj, T_xy)

        # reshape the obj so that it starts with z dim: [batch, x, z, y]
        transposed_transformed_xy = tf.transpose(transformed_xy, [0,2,3,1])

        # transform the z dim 
        s_zx = tf.matmul(w_scale, tf.constant([[1,0,0,0,0,0]],dtype=tf.float32)) + tf.matmul(tf.ones([self.batch_size, 1]), tf.constant([[0,0,0,0,1,0]],dtype=tf.float32))
        t_zx = tf.matmul(w_trans_z, tf.constant([[0,0,1,0,0,0]],dtype=tf.float32))
        T_zx = s_zx + t_zx
        transformed_zxy = stn(transposed_transformed_xy, T_zx)

        # reshape to the original order: [batch, y, x, z]
        transformed_xyz = tf.transpose(transformed_zxy, [0,3,1,2] )

        # add the last dimension back
        transformed_xyz = tf.expand_dims(transformed_xyz, -1)
            
        print(transformed_xyz)

        return transformed_xyz

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

    def leaky_relu(self, x, alpha=0.2):
        negative_part = tf.nn.relu(-x)
        x = tf.nn.relu(x)
        x = x - tf.constant(alpha, dtype=tf.float32) * negative_part
        return x
    
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

    def fc_layer_init(self, inputs, out_dim, name):
        assert len(inputs.get_shape()) == 2
        with tf.name_scope('fc_layer_init'):
            with tf.name_scope('weights'):
                weights = tf.get_variable(name=name + '/weights', dtype=tf.float32, shape=[inputs.get_shape()[1], out_dim], initializer=tf.constant_initializer(0.0))
                tf.summary.histogram(name + '/weights', weights)
            with tf.name_scope('biases'):
                biases = tf.get_variable(name=name + '/biases', shape=[out_dim], dtype=tf.float32, initializer=tf.constant_initializer([1,0,0,0]))
                tf.summary.histogram(name + '/biases', biases)
            with tf.name_scope('fc_out'):
                fc = tf.nn.bias_add(tf.matmul(inputs, weights), biases)
        return fc

    def batch_norm(self, x, is_training, scope):
        return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, 
                                            epsilon=1e-5, scale=True, is_training=is_training, scope=scope)

    def dis(self,o1,o2):
        eucd2 = tf.pow(tf.subtract(o1, o2), 2)
        eucd2 = tf.reduce_sum(eucd2)
        eucd = tf.sqrt(eucd2+1e-6, name='eucd')
        loss = tf.reduce_mean(eucd, name='loss')
        return loss