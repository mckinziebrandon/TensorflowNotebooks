#[8741:READIN]->[[ae90:FULLY_CONNECTED]]
#[ae90:FULLY_CONNECTED]->[[883f:SOFTMAX]]
#[883f:SOFTMAX]->[]

from __future__ import division, print_function, absolute_import

import os
import sys
import tempfile
import urllib
import collections
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.io import arff

import tflearn
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression

def shuffle(df, n=1, axis=0):
    df = df.copy()
    for _ in range(n):
         df.apply(np.random.shuffle, axis=axis)
         return df

def load_data(dfn="/home/joze/Downloads/iclr/data/ailerons.arff"):
        if not os.path.exists(dfn):
                print("Training data not found %s" % dfn)
	Dataset = collections.namedtuple('Dataset', ['data', 'target'])
	Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
	f = open(dfn)
	data, meta = arff.loadarff(f)
	f.close()
	COLUMNS = meta.names()
	label_column = meta.names()[-1]
	CATEGORICAL_COLUMNS = {}
	CONTINUOUS_COLUMNS = []
	for name in meta.names():
		if meta._attributes[name][0] == "numeric":
			CONTINUOUS_COLUMNS.append(name)
		else:
			CATEGORICAL_COLUMNS[name] = len(meta._attributes[name][1])

	data = pd.DataFrame(arff.loadarff(open(dfn))[0], columns=COLUMNS)
	#data=shuffle(data)

        for categorical in CATEGORICAL_COLUMNS:
                data[categorical]=pd.Categorical.from_array(data[categorical]).codes

	lastcolidx=data.columns.values[len(data.columns)-1]
	print("lastcolidx is {}".format(lastcolidx))#remove

	data_class = pd.get_dummies(data[lastcolidx])
	print("names is {}".format(data.columns.values))
	del data[lastcolidx]

	data=pd.get_dummies(data, columns=set(data.columns.values).intersection(CATEGORICAL_COLUMNS))

	columns=len(data.columns)-1 #we remove class, but index not updated: https://github.com/pandas-dev/pandas/issues/2770
	columnssquared=int(math.sqrt(math.pow(int(math.sqrt(columns)),2)))
	dummycols=[]

	print(columns)
	print(columnssquared)

	if((columns - columnssquared)>0):
		increasecolumnscount=int(math.pow(columnssquared+1,2))-columns
		for col in range(1, increasecolumnscount):
    			data['dummycolumn{}'.format(col)] = 0

	size=len(data.index)
	trainsize=0.8
	traindatasetsize=int(size*trainsize)

	train = data[:traindatasetsize]
	train_labels = data_class[:traindatasetsize]
	rest = data[traindatasetsize:]
	rest_labels = data_class[traindatasetsize:]

	restsize=len(rest.index)
	validationsetcount=int(restsize*0.5)
	validation = rest[:validationsetcount]
	validation_labels = rest_labels[:validationsetcount]
	test = rest[validationsetcount:]
	test_labels = rest_labels[validationsetcount:]

	return Datasets(train=Dataset(train.as_matrix(), train_labels.as_matrix()), test=Dataset(test.as_matrix(), test_labels.as_matrix()), validation=Dataset(validation.as_matrix(), validation_labels.as_matrix()))

import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y = shuffle(X, Y)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

out_readin1 = input_data(shape=[None,28,28,1])
out_fully_connected2 = fully_connected(out_readin1, 10)
out_softmax3 = fully_connected(out_fully_connected2, 10, activation='softmax')

network=out_softmax3
hash='f0c188c3777519fb93f1a825ca758a0c'
scriptid='MNIST-f0c188c3777519fb93f1a825ca758a0c'
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')
model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit(X, Y, n_epoch=16, validation_set=(testX, testY), snapshot_step=10, snapshot_epoch=False, show_metric=True, run_id=scriptid)

prediction = model.predict(valX)
auc=roc_auc_score(valY, prediction, average='macro', sample_weight=None)
accuracy=model.evaluate(testX,testY)
print("finalout,%s,%s,%s" % (scriptid,accuracy,auc))

#Test on validation set every n epochs: https://github.com/tflearn/tflearn/issues/176
#Feed dict: http://stackoverflow.com/questions/37267584/tensorflow-feed-dict-using-same-symbol-for-key-value-pair-got-typeerror-can
#Early termination:
# - https://github.com/tflearn/tflearn/issues/361
# - https://github.com/tflearn/tflearn/pull/288
# - http://stackoverflow.com/questions/39751113/early-stopping-with-tflearn/39927599

#val_monitor = tf.contrib.learn.monitors.ValidationMonitor(valX, valY, every_n_steps=50)
