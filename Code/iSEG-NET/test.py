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
FUNC_LABEL_NUM = 18

BATCH_SIZE = 1
MODEL = '../../Model/iSEG/full/model.ckpt'
DATA_IO = '../../DATA/label/labelMat'
DATA_SCE = '../../Output/iGEN/full/gen'
TEST_FILE = '../../DATA/cvlist/test_0.txt'
OUTPUT_FILE = '../../Output/iSEG/full'

if not os.path.exists(OUTPUT_FILE+'/'):
	os.makedirs(OUTPUT_FILE+'/')

cate_label_idx_list = scio.loadmat('../../DATA/label/interactionLabel.mat')
cate_label_idx_list = np.squeeze(cate_label_idx_list['categoryLabelIdx'])

# add noise
def dataAug(input, weight = 1.0):
	dim = input.shape[0]
	c = input.shape[-1]
	data = np.random.rand(dim,dim,dim,c)
	out  = data + input * weight
	f_s = np.sum(out, axis=-1, keepdims=True)
	return out/f_s

print("Loading data...")
dataSce = {}
dataBinary = {}
for model in os.listdir(DATA_SCE):
	data = scio.loadmat(DATA_SCE + '/' + model)
	data = data['instances']
	data = np.reshape(data.data, [64,64,64,3])
	dataSce[model] = data

	data = np.argmax(data,-1)
	binaryData = np.zeros((64,64,64,2))
	free = np.zeros((64,64,64))
	central = np.zeros((64,64,64))
	free[data==2]=1
	central[data==0]=1
	binaryData[:,:,:,0] = central
	binaryData[:,:,:,1] = free
	dataBinary[model] = binaryData

testList = []
f = open(TEST_FILE)
lines = f.readlines()
for line in lines:
	model_x = line.split('\n')[0]+'_rec_sce_4seg.mat'
	temp = np.zeros(25,dtype=np.int32)
	label = line.split('_')[0]
	label_idx_list = []
	label_idx = np.zeros(FUNC_LABEL_NUM+2,dtype=np.int32)
	for i in range(25):
		if CATEGORY_NAME[i] == label:
			temp[i] = 1
			label_idx_list = np.squeeze(cate_label_idx_list[i])
			break
	if label == 'Backpack':
		label_idx_list = [1]
	for idx in label_idx_list:
		label_idx[idx] = 1
	label_idx[0] = 1
	label_idx[-1] = 1
	testList.append([model_x,temp,label_idx])
f.close()
print("dataSce loaded")

# setup network
iSEG = network.iSEG(BATCH_SIZE)
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

x = np.random.rand(BATCH_SIZE, 64, 64, 64, 3)
x_binary = np.random.rand(BATCH_SIZE, 64, 64, 64, 2)
label = np.random.rand(BATCH_SIZE, 25)
label_idx = np.random.rand(BATCH_SIZE, FUNC_LABEL_NUM + 2)
for i in range(len(testList)):
	model_x, label_, label_idx_ = testList[i]
	x[0, :, :, :, :] = dataSce[model_x]
	x_binary[0, :, :, :, :] = dataBinary[model_x]
	label[0] = label_
	label_idx[0] = label_idx_
	result = sess.run([iSEG.x_rec], feed_dict = {iSEG.x: x, iSEG.x_binary:x_binary, iSEG.label:label})
	result = np.reshape(result, [64, 64, 64, FUNC_LABEL_NUM + 2])

	# for refine
	# if not os.path.exists(OUTPUT_FILE+'/refine/'):
	#     os.makedirs(OUTPUT_FILE+'/refine/')
	# scio.savemat(OUTPUT_FILE+'/refine/'+model_x[0:-4]+'_label_refine.mat', {'instances':result})

	# for visual
	label_sce = np.argmax(result,-1)+1
	label_max = np.amax(label_sce)
	label_sce[label_sce==label_max] = 0
	if not os.path.exists(OUTPUT_FILE + '/visual/'):
		os.makedirs(OUTPUT_FILE + '/visual/')
	scio.savemat(OUTPUT_FILE + '/visual/'+model_x[0:-4] + '_label.mat', {'instances':label_sce.astype(np.int8)})