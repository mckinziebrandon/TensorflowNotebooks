#[d440:READIN]->[[9baa:FULLY_CONNECTED]]
#[9baa:FULLY_CONNECTED]->[[5166:SOFTMAX]]
#[5166:SOFTMAX]->[]

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

import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y = shuffle(X, Y)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

out_readin1 = input_data(shape=[None,28,28,1])
out_fully_connected2 = fully_connected(out_readin1, 10)
out_softmax3 = fully_connected(out_fully_connected2, 10, activation='softmax')

validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                        "precision": tf.contrib.metrics.streaming_precision,
                        "recall": tf.contrib.metrics.streaming_recall}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
                                test_set.data,
                                test_set.target,
                                every_n_steps=50,
                                metrics=validation_metrics)

network=out_softmax3
hash='f0c188c3777519fb93f1a825ca758a0c'
scriptid='MNIST-f0c188c3777519fb93f1a825ca758a0c'
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')
model = tflearn.DNN(network, tensorboard_verbose=3)
model.fit(X, Y, n_epoch=1, validation_set=(testX, testY), snapshot_step=10, snapshot_epoch=False, show_metric=True, run_id=scriptid)

prediction = model.predict(testX)
auc=roc_auc_score(testY, prediction, average='macro', sample_weight=None)
accuracy=model.evaluate(testX,testY)

print("Accuracy:", accuracy)
print("ROC AUC Score:", auc)
