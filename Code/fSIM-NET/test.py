import os
import random
import numpy as np
import scipy.io as scio
import tensorflow as tf
import network

BATCH_SIZE = 1
MODEL = '../../Model/fSIM/9-1/model.ckpt'
DATA_SCE = '../../DATA/scene/'
DATA_OBJ = '../../DATA/shape/'
SCE_FILE = '../../DATA/datalist/datalist_scene_train_9-1.txt'
RETRIEVAL_FILE = '../../DATA/datalist/datalist_shape_test_9-1.txt'
OUTPUT_FILE = '../../Output/fSIM/fSIM_9-1_testdata.txt'

fSIM = network.fSIM(BATCH_SIZE)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
tf.global_variables_initializer().run()

model_ckpt = MODEL + '.index'
if os.path.isfile(model_ckpt):
	print("model.ckpt file found")
else:
	print("model file not found, quit.")
	exit()
saver = tf.train.Saver()
saver.restore(sess, MODEL)
print("Model restored")

# Load data
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

sce_list = []
f = open(SCE_FILE)
for line in f.readlines():
	sce_list.append(line.split(' ')[0])
f.close()
print("Scene list loaded")

retrieval_list = []
f = open(RETRIEVAL_FILE)
for line in f.readlines():
	retrieval_list.append(line.split(' ')[0])
f.close()
print("Object list loaded")

sce_feature = []
batch_y = np.random.rand(1, 64, 64, 64, 3)
for i in range(len(sce_list)):
	batch_y[0, :, :, :, :] = dataMap[sce_list[i]]
	out = sess.run([fSIM.out_y], feed_dict={fSIM.in_y: batch_y})
	sce_feature.append(out[0][0])
print("Scene feature got")

# -----Calculate the probability for retrieval objects-----
f = open(OUTPUT_FILE, 'w')
mu1, mu2, mu3, stddev1, stddev2, stddev3, weight1, weight2, weight3 = ([] for i in range(9))
batch_x = np.random.rand(1, 64, 64, 64, 1)
for k in range(len(retrieval_list)):
	batch_x[0, :, :, :, :] = dataMap[retrieval_list[k]]
	w1, w2, w3, m1, m2, m3, s1, s2, s3 = sess.run(
		[fSIM.weight1, fSIM.weight2, fSIM.weight3, fSIM.mu1, fSIM.mu2, fSIM.mu3,
		fSIM.stddev1, fSIM.stddev2, fSIM.stddev3], feed_dict={fSIM.in_x: batch_x})
	weight1.append(w1)
	weight2.append(w2)
	weight3.append(w3)
	mu1.append(m1)
	mu2.append(m2)
	mu3.append(m3)
	stddev1.append(s1)
	stddev2.append(s2)
	stddev3.append(s3)
print("GMM para got\n")

prob_tensor, prob, result = [], [], []
W1, W2, W3, M1, M2, M3, S1, S2, S3, F = (tf.placeholder(tf.float32) for i in range(10))
calculate_prob = tf.log(tf.contrib.distributions.MultivariateNormalDiag(M1, S1).prob(F) * W1 +
						tf.contrib.distributions.MultivariateNormalDiag(M2, S2).prob(F) * W2 +
						tf.contrib.distributions.MultivariateNormalDiag(M3, S3).prob(F) * W3 + 1.0e-10)
for k in range(len(retrieval_list)):
	print("Retrieval object:", retrieval_list[k])
	f.write("Retrieval: " + str(retrieval_list[k])[:-4] + " \n")
	for i in range(len(sce_feature)):
		prob.append(sess.run(calculate_prob, feed_dict={
					W1: weight1[k], W2: weight2[k], W3: weight3[k], M1: mu1[k], M2: mu2[k], M3: mu3[k],
					S1: stddev1[k], S2: stddev2[k], S3: stddev3[k], F: sce_feature[i]}))
	for m in range(len(sce_feature)):
		result.append(prob[m][0][0])
		f.write(str(sce_list[m])[:-4] + " " + str(result[m]) + " \n")

	# ---------find scene with max probability thus get the label--------
	prob_max = max(result)
	max_indice = result.index(max(result))-1
	print("MAX scene:", sce_list[max_indice], "prob:", prob_max, "\n")
	f.write("MAX scene: " + str(sce_list[max_indice]) + " prob: " + str(prob_max) + "\n\n")

	f.write("\n")
	prob = []
	result = []
f.close()