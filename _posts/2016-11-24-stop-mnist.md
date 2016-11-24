---
title: Stopping CodeGenerated MNIST with TFLearn
layout: post
---

## Current Issues

The main problem is that the training state object that provides the validation accuracy info to the callback object is not
storing the validation accuracy in its instance variables. I've debugged it around in circles and this is the main thing
preventing early stopping from working properly. I've dug through the tflearn source code and it looks like this value should get
stored. It is most likely related to whatever default trainOp gets passed to the DNN class.


```python
from __future__ import division, print_function, absolute_import

import os
import sys
import tempfile
import urllib
import collections
import math

import numpy as np
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
```

    hdf5 not supported (please install/reinstall h5py)



```python
X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y  = shuffle(X, Y)
X     = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])
```

    Extracting mnist/train-images-idx3-ubyte.gz
    Extracting mnist/train-labels-idx1-ubyte.gz
    Extracting mnist/t10k-images-idx3-ubyte.gz
    Extracting mnist/t10k-labels-idx1-ubyte.gz



```python
out_readin1          = input_data(shape=[None,28,28,1])
out_fully_connected2 = fully_connected(out_readin1, 10)
out_softmax3         = fully_connected(out_fully_connected2, 10, activation='softmax')

hash='f0c188c3777519fb93f1a825ca758a0c'
scriptid='MNIST-f0c188c3777519fb93f1a825ca758a0c'

network = regression(out_softmax3, 
                     optimizer='adam', 
                     learning_rate=0.01, 
                     loss='categorical_crossentropy', 
                     name='target')

#model = tflearn.DNN(network, tensorboard_verbose=3)
```


```python
model.fit(X, Y, 
          n_epoch=1, 
          validation_set=(testX, testY), 
          snapshot_step=10, 
          snapshot_epoch=False, 
          show_metric=True, 
          run_id=scriptid)


prediction = model.predict(testX)
auc=roc_auc_score(testY, prediction, average='macro', sample_weight=None)
accuracy=model.evaluate(testX,testY)

print("Accuracy:", accuracy)
print("ROC AUC Score:", auc)
```

# Now Trying with TFLearn

**Issue**: I can't seem to figure this out for the life of me, but for some reason training_state never has a non-none value for val_acc, and essentially most other evaluation metrics. I'm assuming this is because I need to explicitly tell TFLearn to store them every n iterations, but the documentation suggests that the default behavior is to store these basic values. 


```python
import pdb
class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        self.val_acc_thresh = val_acc_thresh
    
    def on_epoch_end(self, training_state):
        """ """
        # Apparently this can happen.
        pdb.set_trace()
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration
            
    def on_batch_end(self, training_state, snapshot=False):
        """ """
        # Apparently this can happen.
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration
            
            
# Initializae our callback.
early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.5)

model = tflearn.DNN(network, tensorboard_verbose=1)
# Give it to our trainer and let it fit the data. 
model.fit(X[:20000], Y[:20000], 
          n_epoch=3, 
          validation_set=(testX, testY), 
          snapshot_epoch=True, 
          #show_metric=True, 
          callbacks=early_stopping_cb)
```

    Training Step: 313  | total loss: [1m[32m0.34588[0m[0m
    | Adam | epoch: 001 | loss: 0.34588 | val_loss: 0.34831 -- iter: 20000/20000
    Training Step: 313  | total loss: [1m[32m0.34588[0m[0m
    | Adam | epoch: 001 | loss: 0.34588 | val_loss: 0.34831 -- iter: 20000/20000
    --
    > <ipython-input-4-62e7ee0640e6>(11)on_epoch_end()
    -> if training_state.val_acc is None: return
    (Pdb) training_state.val_acc
    (Pdb) training_state.global_acc
    (Pdb) training_state.acc_value
    (Pdb) training_state.loss_value
    0.34587910771369934



```python

```
