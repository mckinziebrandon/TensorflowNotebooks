
# Now Trying with TFLearn

## It Works! Here's How. 


The following is a code snippet directly from [trainer.py](https://github.com/tflearn/tflearn/blob/master/tflearn/helpers/trainer.py#L281) in the tflearn github repository, where I'm only showing the relevant parts/logic. 

```python
try:
    for epoch in range(n_epoch):
        # . . . Setup stuff for epoch here . . . 
        for batch_step in range(max_batches_len):
            # . . . Setup stuff for next batch here . . . 
            for i, train_op in enumerate(self.train_ops):
                caller.on_sub_batch_begin(self.training_state)
                
                # Train our model and store desired information in the train_op that
                # we (the user) pass to the trainer as an initialization argument.
                snapshot = train_op._train(self.training_state.step,
                                           (bool(self.best_checkpoint_path) | snapshot_epoch),
                                           snapshot_step,
                                           show_metric)
                                           
                # Update training state. The training state object tells us 
                # how our model is doing at various stages of training.
                self.training_state.update(train_op, train_ops_count)

            # All optimizers batch end
            self.session.run(self.incr_global_step)
            caller.on_batch_end(self.training_state, snapshot)

        # ---------- [What we care about] -------------
        # Epoch end. We define what on_epoch_end does. In this
        # case, I'll have it raise an exception if our validation accuracy
        # reaches some desired threshold. 
        caller.on_epoch_end(self.training_state)
        # ---------------------------------------------

finally:
    # Once we raise the exception, this code block will execute. 
    # Note only afterward will our catch block execute. 
    caller.on_train_end(self.training_state)
    for t in self.train_ops:
        t.train_dflow.interrupt()
    # Set back train_ops
    self.train_ops = original_train_ops
```


## Setup the Basic Network Architecture


```python
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

n_features = 784
n_hidden = 256
n_classes = 10

# Define the inputs/outputs/weights as usual.
X = tf.placeholder("float", [None, n_features])
Y = tf.placeholder("float", [None, n_classes])

# Define the connections/weights and biases between layers.
W1 = tf.Variable(tf.random_normal([n_features, n_hidden]), name='W1')
W2 = tf.Variable(tf.random_normal([n_hidden, n_hidden]), name='W2')
W3 = tf.Variable(tf.random_normal([n_hidden, n_classes]), name='W3')

b1 = tf.Variable(tf.random_normal([n_hidden]), name='b1')
b2 = tf.Variable(tf.random_normal([n_hidden]), name='b2')
b3 = tf.Variable(tf.random_normal([n_classes]), name='b3')

# Define the operations throughout the network.
net = tf.tanh(tf.add(tf.matmul(X, W1), b1))
net = tf.tanh(tf.add(tf.matmul(net, W2), b2))
net = tf.add(tf.matmul(net, W3), b3)


# Define the optimization problem.
loss      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
accuracy  = tf.reduce_mean(tf.cast(
        tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1) ), tf.float32), name='acc')
```

    hdf5 not supported (please install/reinstall h5py)
    Extracting mnist/train-images-idx3-ubyte.gz
    Extracting mnist/train-labels-idx1-ubyte.gz
    Extracting mnist/t10k-images-idx3-ubyte.gz
    Extracting mnist/t10k-labels-idx1-ubyte.gz


## Define the TrainOp and Trainer Objects


```python
trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, metric=accuracy, batch_size=128)
trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=1)
```

# The EarlyStoppingCallback Class

I show a proof-of-concept version of early stopping below. This is the simplest possible case: just stop training after the first epoch no matter what. It is up to the user to decide the conditions they want to trigger the stopping on.


```python
import tflearn
class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        # Store a validation accuracy threshold, which we can compare against
        # the current validation accuracy at, say, each epoch, each batch step, etc.
        self.val_acc_thresh = val_acc_thresh
    
    def on_epoch_end(self, training_state):
        """ 
        This is the final method called in trainer.py in the epoch loop. 
        We can stop training and leave without losing any information with a simple exception.  
        """
        print("Terminating training at the end of epoch", training_state.epoch)
        raise StopIteration
    
    def on_train_end(self, training_state):
        """
        Furthermore, tflearn will then immediately call this method after we terminate training, 
        (or when training ends regardless). This would be a good time to store any additional 
        information that tflearn doesn't store already.
        """
        print("Successfully left training! Final model accuracy:", training_state.acc_value)
       
        
# Initialize our callback with desired accuracy threshold.  
early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.5)
```

    hdf5 not supported (please install/reinstall h5py)


# Result: Train the Model and Stop Early


```python
try:
    # Give it to our trainer and let it fit the data. 
    trainer.fit(feed_dicts={X: trainX, Y: trainY}, 
                val_feed_dicts={X: testX, Y: testY}, 
                n_epoch=1, 
                show_metric=True, # Calculate accuracy and display at every step.
                callbacks=early_stopping_cb)
except StopIteration:
    print("Caught callback exception. Returning control to user program.")
    
```

    Training Step: 860  | total loss: [1m[32m1.73372[0m[0m
    | Optimizer | epoch: 002 | loss: 1.73372 - acc: 0.8196 | val_loss: 1.87058 - val_acc: 0.8011 -- iter: 55000/55000
    Training Step: 860  | total loss: [1m[32m1.73372[0m[0m
    | Optimizer | epoch: 002 | loss: 1.73372 - acc: 0.8196 | val_loss: 1.87058 - val_acc: 0.8011 -- iter: 55000/55000
    --
    Terminating training at the end of epoch 2
    Successfully left training! Final model accuracy: 0.8196054697036743
    Caught callback exception. Returning control to user program.


# Appendix

For my own reference, this is the code I started with before tinkering with the early stopping solution above.


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


# Load the data and handle any preprocessing here.
X, Y, testX, testY = mnist.load_data(one_hot=True)
X, Y  = shuffle(X, Y)
X     = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

# Define our network architecture: a simple 2-layer network of the form
# InputImages -> Fully Connected -> Softmax
out_readin1          = input_data(shape=[None,28,28,1])
out_fully_connected2 = fully_connected(out_readin1, 10)
out_softmax3         = fully_connected(out_fully_connected2, 10, activation='softmax')

hash='f0c188c3777519fb93f1a825ca758a0c'
scriptid='MNIST-f0c188c3777519fb93f1a825ca758a0c'

# Define our training metrics. 
network = regression(out_softmax3, 
                     optimizer='adam', 
                     learning_rate=0.01, 
                     loss='categorical_crossentropy', 
                     name='target')

model = tflearn.DNN(network, tensorboard_verbose=3)
try:
    model.fit(X, Y, n_epoch=1, validation_set=(testX, testY), 
          snapshot_epoch=False, 
          show_metric=True, 
          run_id=scriptid,callbacks=early_stopping_cb)
except StopIteration:
    print("Caught callback exception. Returning control to user program.")


prediction = model.predict(testX)
auc=roc_auc_score(testY, prediction, average='macro', sample_weight=None)
accuracy=model.evaluate(testX,testY)

print("Accuracy:", accuracy)
print("ROC AUC Score:", auc)

```

    Training Step: 860  | total loss: [1m[32m0.30941[0m[0m
    | Adam | epoch: 001 | loss: 0.30941 - acc: 0.9125 -- iter: 55000/55000
    Terminating training at the end of epoch 1
    Successfully left training! Final model accuracy: 0.9125033020973206
    Caught callback exception. Returning control to user program.
    Accuracy: [0.90410000000000001]
    ROC AUC Score: 0.992379719297



```python

```