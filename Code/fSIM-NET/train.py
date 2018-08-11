import os
import random
import numpy as np
import scipy.io as scio
import tensorflow as tf
import network

EPOCHS = 15000
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
DATA_OBJ = '../../DATA/shape'
DATA_SCE = '../../DATA/scene'		
TRAIN_FILE = './datalist/train_9-1_25cate.txt'
MODEL_PATH = '../../Model/fSIM/9-1'
LOG_PATH = '../../Log/fSIM/9-1'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

print("Loading data...")
dataMap = {}
for model in os.listdir(DATA_OBJ):
	data = scio.loadmat(DATA_OBJ + '/' + model)
	data = data['instances']
	data = np.reshape(data.data, [64,64,64,1])
	dataMap[model] = data
for model in os.listdir(DATA_SCE):
	data = scio.loadmat(DATA_SCE + '/' + model)
	data = data['instances']
	data = np.reshape(data.data, [64,64,64])
	newData = np.zeros((64,64,64,3))
	free = np.zeros((64,64,64))
	inter = np.zeros((64,64,64))
	central = np.zeros((64,64,64))
	free[data==0]=1
	central[data==1]=1
	inter[data==2]=1
	newData[:,:,:,0] = central
	newData[:,:,:,1] = inter
	newData[:,:,:,2] = free
	dataMap[model] = newData
print("Dataset loaded:", len(dataMap), "models")

trainList = []
f = open(TRAIN_FILE)
lines = f.readlines()
for line in lines:
	trainList.append(line.split(' ')[:-1])
random.shuffle(trainList)
f.close()

# Setup network
fSIM = network.fSIM(BATCH_SIZE)
with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(fSIM.loss)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))	
tf.global_variables_initializer().run()

saver = tf.train.Saver()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

new = True
model_ckpt = MODEL_PATH +'/model.ckpt.index'
if os.path.isfile(model_ckpt):
	input_var = None
	while input_var not in ['yes', 'no']:
		input_var = input("model.ckpt file found. Do you want to quit [yes/no]?")
	if input_var == "yes":
		new = False
if new:
	print("start training...")
	trainLen = len(trainList)
	for epoch in range(EPOCHS):
		batch_xs = np.random.rand(BATCH_SIZE, 64, 64, 64, 1)
		batch_ys = np.random.rand(BATCH_SIZE, 64, 64, 64, 3)
		batch_zs = np.random.rand(BATCH_SIZE, 64, 64, 64, 3)
		for i in range(BATCH_SIZE):
			model_x, model_y, model_z = trainList[(epoch * BATCH_SIZE + i) % trainLen]
			batch_xs[i,:,:,:,:] = dataMap[model_x]
			batch_ys[i,:,:,:,:] = dataMap[model_y]
			batch_zs[i,:,:,:,:] = dataMap[model_z]
		_, loss = sess.run([optimizer, fSIM.loss], feed_dict={
							fSIM.in_x: batch_xs, fSIM.in_y: batch_ys, fSIM.in_z: batch_zs})
		if epoch % 10 == 0:
			print("epoch: %d loss: %f" % (epoch, loss))
			train_log = sess.run(merged, feed_dict={
								 fSIM.in_x: batch_xs, fSIM.in_y: batch_ys, fSIM.in_z: batch_zs})
			writer.add_summary(train_log, epoch)	
	saver.save(sess, MODEL_PATH+'/model.ckpt')
	print("Model saved")
else:
	exit()
