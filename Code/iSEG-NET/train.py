import tensorflow as tf
import numpy as np
import scipy.io as scio
import os
import sys
import random
import network

CATEGORY_NAME = ['Backpack', 'Basket', 'Bathtub', 'Bed', 'Bench', 'Bicycle', 'Bowl', 'Chair', 'Cup', 'Desk', 'DryingRack', 'Handcart',
               'Hanger', 'Hook', 'Lamp', 'Laptop', 'Shelf', 'Sink', 'Sofa', 'Stand', 'Stool', 'Stroller', 'Table', 'Tvbench', 'Vase']
FUNC_LABEL_NUM = 18
AUG = 5

EPOCHS = 3000
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
DATA_IO = '../../DATA/label/labelMat'
DATA_SCE = '../../DATA/scene'
TRAIN_FILE = '../../DATA/datalist/datalist_scene_full_25cate.txt'
MODEL_PATH = '../../Model/iSEG/full'
LOG_PATH = '../../Log/iSEG/full'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=-1,keepdims=True)

def dataAug(input, weight = 1.0):
    dim = input.shape[0]
    c = input.shape[-1]
    data = np.random.rand(dim,dim,dim,c)
    out  = data + input * weight
    f_s = np.sum(out, axis=-1, keepdims=True)
    return out/f_s # softmax(out)

cate_label_idx_list = scio.loadmat('../../DATA/label/interactionLabel.mat')
cate_label_idx_list = np.squeeze(cate_label_idx_list['categoryLabelIdx'])

# data prepare
print("Loading data...")
dataSce = {}
dataBinary = {}
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

    binaryData = np.zeros((64,64,64,2))
    binaryData[:,:,:,0] = newData[:,:,:,0]
    binaryData[:,:,:,1] = newData[:,:,:,2]
    for num in range(AUG):
        augModel = model.split('.')[0]+'_'+str(num)
        dataSce[augModel] = dataAug(newData)
        dataBinary[augModel] = binaryData
print("dataSce loaded")

# ground truth
dataIO = {}
for model in os.listdir(DATA_IO):
    if 'label' in model:
        data = scio.loadmat(DATA_IO + '/' + model)
        data = data['instances']
        data = np.reshape(data.data, [64,64,64])
        newData = np.zeros((64,64,64,FUNC_LABEL_NUM+2))
        central = np.zeros((64,64,64))
        free = np.zeros((64,64,64))
        label = model.split('_')[0]
        label_idx = []
        for i in range(25):
            if CATEGORY_NAME[i] == label:
                label_idx = np.squeeze(cate_label_idx_list[i])
            if label == 'Backpack':
                label_idx = [1]
        free[data==0]=1
        central[data==1]=1
        newData[:,:,:,0] = central
        newData[:,:,:,-1] = free
        for i in range(len(label_idx)):
            temp = np.zeros((64,64,64))
            temp[data==i+2]=1   # interaction label begin at 2
            newData[:,:,:,label_idx[i]] = temp  # interaction channel begin at channel 1
        for num in range(AUG):
            augModel = model.split('.')[0][0:-6]+'_'+str(num)
            dataIO[augModel] = newData
print("data loaded")

trainList = []
f = open(TRAIN_FILE)
lines = f.readlines()
for line in lines:
    model_x = line.split(' \n')[0]
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
    for num in range(AUG):
        augModel = model_x.split('.')[0]+'_'+str(num)
        trainList.append([augModel,temp,label_idx])
random.shuffle(trainList)
f.close()

# setup network
iSEG = network.iSEG(BATCH_SIZE)
with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, name='Adam').minimize(iSEG.loss)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))	
tf.global_variables_initializer().run()

saver = tf.train.Saver(max_to_keep=50) 
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

new = True
model_ckpt = MODEL_PATH+'.index'
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        input_var = input("We found model.ckpt file. Do you want to quit [yes/no]?")
    if input_var == 'yes':
        new = False
if new:
    x_io = np.random.rand(BATCH_SIZE, 64, 64, 64, FUNC_LABEL_NUM+2)
    x_binary = np.random.rand(BATCH_SIZE, 64, 64, 64, 2)
    x = np.random.rand(BATCH_SIZE, 64, 64, 64, 3)
    label = np.random.rand(BATCH_SIZE, 25)
    label_idx = np.random.rand(BATCH_SIZE, FUNC_LABEL_NUM+2)
    trainLen = len(trainList)
    print('Begin training')
    for epoch in range(EPOCHS):
        for i in range(BATCH_SIZE):
            model_x, label_, label_idx_ = trainList[(epoch * BATCH_SIZE + i) % trainLen]
            x[i, :, :, :, :] = dataSce[model_x]
            x_io[i, :, :, :, :] = dataIO[model_x]
            x_binary[i, :, :, :, :] = dataBinary[model_x]
            label[i] = label_
            label_idx[i] = label_idx_
        _, loss = sess.run([optimizer, iSEG.loss], feed_dict = {iSEG.x: x, iSEG.x_io: x_io, 
                                                                iSEG.x_binary:x_binary, iSEG.label:label, iSEG.label_idx:label_idx})
        if epoch%10 == 0:
            print ('iter :', epoch, 'loss', loss)
        train_log = sess.run(merged, feed_dict = {iSEG.x: x, iSEG.x_io: x_io})
        writer.add_summary(train_log, epoch)
        
    saver.save(sess, MODEL_PATH + '/model.ckpt')
    print('Model saved')
else:
    quit()