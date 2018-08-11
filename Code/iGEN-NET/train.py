import tensorflow as tf
import numpy as np
import scipy.io as scio
import os
import sys
import random
import network

CATEGORY_NAME = ['Backpack', 'Basket', 'Bathtub', 'Bed', 'Bench', 'Bicycle', 'Bowl', 'Chair', 'Cup', 'Desk', 'DryingRack', 'Handcart',
               'Hanger', 'Hook', 'Lamp', 'Laptop', 'Shelf', 'Sink', 'Sofa', 'Stand', 'Stool', 'Stroller', 'Table', 'Tvbench', 'Vase']
is_training = True

EPOCHS = 6000
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
DATA_OBJ = '../../DATA/shape'
DATA_SCE = '../../DATA/scene'
TRANS_FILE = './datalist/trans'
TRANS_NAME_LIST = './datalist/scene_list'
TRAIN_FILE = '../../DATA/datalist/datalist_scene_full_25cate.txt'
MODEL_PATH = '../../Model/iGEN/full'
LOG_PATH = '../../Log/iGEN/full'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
print(MODEL_PATH)

# data prepare
dataSce = {} ## channel order
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
    dataSce[model] = newData
print("dataSce loaded")

dataObj = {}
for model in os.listdir(DATA_OBJ):
	data = scio.loadmat(DATA_OBJ + '/' + model)
	data = data['instances']
	data = np.reshape(data.data, [64,64,64,1])
	dataObj[model] = data
print('dataObj loaded')

dataTrans = {}
for cate_name in CATEGORY_NAME:
    trans = scio.loadmat(TRANS_FILE+'/'+cate_name+'_center_scene_trans.mat')
    data = trans['t']
    lines = open(TRANS_NAME_LIST+'/'+cate_name+'.txt').readlines()
    for i in range(len(data)):
        modelname = lines[i].split(' ')[0]
        dataTrans[modelname] = data[i,:]
print('dataTrans loaded')

trainList = []
f = open(TRAIN_FILE)
lines = f.readlines()
for line in lines:
    x_obj = line.split('.')[0]+'_01.mat'
    x_sce = line.split(' ')[0]
    temp = np.zeros(25,dtype=np.int32)
    label = line.split('_')[0]
    for i in range(25):
        if CATEGORY_NAME[i] == label:
            temp[i] = 1
            trainList.append([x_sce,x_obj,temp])
            break
random.shuffle(trainList)
f.close()

para_ts = [var for var in tf.trainable_variables() if any(x in var.name for x in ['_e', 'fc_w', 'encoder'])]
para_rec = [var for var in tf.trainable_variables() if any(x in var.name for x in ['_d', 'scale'])]

# setup network
iGEN = network.iGEN(BATCH_SIZE,is_training)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    with tf.name_scope('train'):
        optimizer_ts = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(iGEN.ts_loss, var_list=para_ts)
        optimizer_rec = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(iGEN.rec_loss, var_list=para_rec)
        optimizer_all = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(iGEN.loss)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
tf.global_variables_initializer().run()

saver = tf.train.Saver(max_to_keep=50)
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
    x_sce = np.random.rand(BATCH_SIZE, 64, 64, 64, 3)
    x_obj = np.random.rand(BATCH_SIZE, 64, 64, 64, 1)
    label = np.random.rand(BATCH_SIZE, 25)
    scale = np.random.rand(BATCH_SIZE)
    trans = np.random.rand(BATCH_SIZE, 3)
    trainLen = len(trainList)
    print('Begin training')
    for epoch in range(EPOCHS):
        for i in range(BATCH_SIZE):
            x_sce_, x_obj_, label_ = trainList[(epoch * BATCH_SIZE + i) % trainLen]
            x_sce[i, :, :, :, :] = dataSce[x_sce_]
            x_obj[i, :, :, :, :] = dataObj[x_obj_]
            label[i] = label_
            scale[i] = float(dataTrans[x_sce_][0])
            trans[i][0] = float(dataTrans[x_sce_][1])
            trans[i][1] = float(dataTrans[x_sce_][2])
            trans[i][2] = float(dataTrans[x_sce_][3])
        if epoch < 2500:
            _, ts_loss, rec_loss, loss = sess.run([optimizer_ts, iGEN.ts_loss, iGEN.rec_loss, iGEN.loss], 
                    feed_dict = {iGEN.x_sce: x_sce, iGEN.x_obj: x_obj, iGEN.label: label, iGEN.S:scale ,iGEN.T:trans})
        elif epoch < 5000:
            _, ts_loss, rec_loss, loss = sess.run([optimizer_rec, iGEN.ts_loss, iGEN.rec_loss, iGEN.loss], 
                    feed_dict = {iGEN.x_sce: x_sce, iGEN.x_obj: x_obj, iGEN.label: label, iGEN.S:scale ,iGEN.T:trans})
        else:
            _, ts_loss, rec_loss, loss = sess.run([optimizer_all, iGEN.ts_loss, iGEN.rec_loss, iGEN.loss], 
                    feed_dict = {iGEN.x_sce: x_sce, iGEN.x_obj: x_obj, iGEN.label: label, iGEN.S:scale ,iGEN.T:trans})

        if epoch % 10 == 0:    
            print ('iter :', epoch, 'ts_loss', ts_loss, 'rec_loss', rec_loss, 'loss', loss)
            train_log = sess.run(merged, feed_dict = {iGEN.x_sce: x_sce, iGEN.x_obj: x_obj, iGEN.label: label, iGEN.S:scale ,iGEN.T:trans})
            writer.add_summary(train_log, epoch)

    saver = tf.train.Saver()
    saver.save(sess, MODEL_PATH+'/model.ckpt')
    print(epoch,'Model saved')
else:
	exit()