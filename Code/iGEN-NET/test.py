import tensorflow as tf
import numpy as np
import scipy.io as scio
import os
import sys
import random
import network
import math

CATEGORY_NAME = ['Backpack', 'Basket', 'Bathtub', 'Bed', 'Bench', 'Bicycle', 'Bowl', 'Chair', 'Cup', 'Desk', 'DryingRack', 'Handcart',
				 'Hanger', 'Hook', 'Lamp', 'Laptop', 'Shelf', 'Sink', 'Sofa', 'Stand', 'Stool', 'Stroller', 'Table', 'Tvbench', 'Vase']
is_training = False

BATCH_SIZE = 64
MODEL = '../../Model/iGEN/full/model.ckpt'
DATA_OBJ = '../../DATA/shape'
TEST_FILE = '../../DATA/cvlist/test_0.txt'
OUTPUT_FILE = '../../Output/iGEN/full'

dataObj = {}
for model in os.listdir(DATA_OBJ):
	data = scio.loadmat(DATA_OBJ + '/' + model)
	data = data['instances']
	data = np.reshape(data.data, [64,64,64,1])
	dataObj[model] = data
print('dataObj loaded')

testList = []
f = open(TEST_FILE)
lines = f.readlines()
for line in lines:
	x_obj = line.split('\n')[0]+'_01.mat'
	x_sce = line.split('\n')[0]
	temp = np.zeros(25,dtype=np.int32)
	label = line.split('_')[0]
	for i in range(25):
		if CATEGORY_NAME[i] == label:
			temp[i] = 1
			testList.append([x_sce,x_obj,temp])
			break
random.shuffle(testList)
f.close()

# setup network
iGEN = network.iGEN(BATCH_SIZE, is_training)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
tf.global_variables_initializer().run()

model_ckpt = MODEL + '.index'
print(MODEL)
if os.path.isfile(model_ckpt):
	print('model.ckpt file found')
else:
	print('model file not found, quit.')
	exit()
saver = tf.train.Saver()
saver.restore(sess, MODEL)
print('Model restored')

x_sce = np.random.rand(BATCH_SIZE, 64, 64, 64, 3)
x_obj = np.random.rand(BATCH_SIZE, 64, 64, 64, 1)
label = np.random.rand(BATCH_SIZE, 25)
testLen = len(testList)
print('Begin testing')
for j in range(int(math.ceil(testLen*1.0/BATCH_SIZE))):
	for i in range(BATCH_SIZE):
		x_sce_, x_obj_, label_ = testList[(j*BATCH_SIZE+i) % len(testList)]
		x_obj[i, :, :, :, :] = dataObj[x_obj_]
		label[i] = label_
	result_obj, result_sce = sess.run([iGEN.cobj,iGEN.x_rec], feed_dict = {iGEN.x_obj: x_obj, iGEN.label: label})

	for i in range(BATCH_SIZE):
		x_obj_name = testList[(j*BATCH_SIZE+i) % len(testList)][1]
		x_sce_name = testList[(j*BATCH_SIZE+i) % len(testList)][0]

		# visualize
		if not os.path.exists(OUTPUT_FILE+'/visual/'):
			os.mkdir(OUTPUT_FILE+'/visual/')
		rec_sce = result_sce[i,:,:,:]
		rec_sce = np.argmax(rec_sce,-1)+1
		rec_sce[rec_sce==3] = 0
		scio.savemat(OUTPUT_FILE+'/visual/'+x_sce_name+'_rec_sce.mat', {'instances':rec_sce.astype(np.int8)})

		# reconstructed scene x
		# multi channel for segmentation
		# if not os.path.exists(OUTPUT_FILE+'/gen/'):
		#     os.mkdir(OUTPUT_FILE+'/gen/')
		# scio.savemat(OUTPUT_FILE+'/gen/'+x_sce_name+'_rec_sce_4seg.mat', {'instances':result_sce[i,:,:,:,:]})

		print(x_sce_name,'saved')
sess.close()