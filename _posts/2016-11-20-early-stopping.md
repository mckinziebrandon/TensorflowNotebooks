---
title: Early Stopping with TensorFlow and TFLearn
layout: post
---

```python
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

trainX, trainY, testX, testY = mnist.load_data(one_hot=True)
```

    hdf5 not supported (please install/reinstall h5py)
    Extracting mnist/train-images-idx3-ubyte.gz
    Extracting mnist/train-labels-idx1-ubyte.gz
    Extracting mnist/t10k-images-idx3-ubyte.gz
    Extracting mnist/t10k-labels-idx1-ubyte.gz



```python
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

# Early Stopping

## Training Setup

In tflearn, we can train our model with a [tflearn.Trainer](http://tflearn.org/helpers/trainer/ "Documentation") object: "Generic class to handle any TensorFlow graph training. It requires the use of TrainOp to specify all optimization parameters."

* [TrainOp](http://tflearn.org/helpers/trainer/#trainop) represents a set of operation used for optimizing a network.

* __Example__: Time to initialize our trainer to work with our MNIST network. Below we create a TrainOp object that is then used for the purpose of telling our trainer 
    1. Our loss function. (softmax cross entropy with logits)
    2. Our optimizer. (GradientDescentOptimizer)
    3. Our evaluation [tensor] metric. (classification accuracy)  


```python
trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, metric=accuracy, batch_size=128)
trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=1)
```

## Callbacks

The [Callbacks](http://tflearn.org/getting_started/#training-callbacks) interface describes a set of methods that we can implement ourselves that will be called during runtime. Below are our options, where here we will be primarily concerned with the on_epoch_end() method.
* __ Methods __ :

```python
    def on_train_begin(self, training_state):
    def on_epoch_begin(self, training_state):
    def on_batch_begin(self, training_state):
    def on_sub_batch_begin(self, training_state):
    def on_sub_batch_end(self, training_state, train_index=0):
    def on_batch_end(self, training_state, snapshot=False):
    def on_epoch_end(self, training_state):
    def on_train_end(self, training_state):
```

* __TrainingState__: Notice that each method requires us to pass a [training_state](https://github.com/tflearn/tflearn/blob/master/tflearn/helpers/trainer.py#L971) object as an argument. These useful helpers will be able to provide us with the information we need to determine when to stop training. Below is a list of the instance variables we can access with a training_state object:
    * self.epoch
    * self.step
    * self.current_iter 
    * self.acc_value 
    * self.loss_value
    * self.val_acc
    * self.val_loss
    * self.best_accuracy
    * self.global_acc
    * self.global_loss
    
* __Implementing our Callback__: Let's say we want to stop training when the validation accuracy reaches a certain threshold. Below, we implement the code required to define such a callback and fit the MNIST data. 


```python
class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        """ Note: We are free to define our init function however we please. """
        self.val_acc_thresh = val_acc_thresh
    
    def on_epoch_end(self, training_state):
        """ """
        # Apparently this can happen.
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            raise StopIteration
```


```python
# Initializae our callback.
early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.5)
# Give it to our trainer and let it fit the data. 
trainer.fit(feed_dicts={X: trainX, Y: trainY}, 
            val_feed_dicts={X: testX, Y: testY}, 
            n_epoch=2, 
            show_metric=True, # Calculate accuracy and display at every step.
            snapshot_epoch=False,
            callbacks=early_stopping_cb)
```

    Training Step: 1720  | total loss: [1m[32m0.81290[0m[0m
    | Optimizer | epoch: 004 | loss: 0.81290 - acc_2: 0.8854 -- iter: 55000/55000


# Using tf.contrib.learn instead

## Iris data loading/tutorial prep

Note: can also load via:
```python
import csv
import random
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)
iris = datasets.load_iris()
print(iris.data.shape)
print("Xt", X_train.shape, "Yt", y_train.shape)
```


```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Suppress the massive amount of warnings.
tf.logging.set_verbosity(tf.logging.ERROR)

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING,
                                                       target_dtype=np.int, 
                                                        features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST,
                                                   target_dtype=np.int, 
                                                    features_dtype=np.float32)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model")

# Fit model.
classifier.fit(x=X_train,
               y=y_train,
               steps=2000)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=X_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))
```

    Accuracy: 0.980000
    Predictions: [1 1]


## Validation Monitors


```python
# Vanilla version
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(test_set.data,
                                                                 test_set.target,
                                                                 every_n_steps=50)

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model",
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=1))

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
               monitors=[validation_monitor])
```




    Estimator(params={'dropout': None, 'hidden_units': [10, 20, 10], 'weight_column_name': None, 'feature_columns': [_RealValuedColumn(column_name='', dimension=4, default_value=None, dtype=tf.float32, normalizer=None)], 'optimizer': 'Adagrad', 'n_classes': 3, 'activation_fn': <function relu at 0x7f8568caa598>, 'num_ps_replicas': 0, 'gradient_clip_norm': None, 'enable_centered_bias': True})



## Customizing the Evaluation Metrics and Stopping Early

If we run the code below, it stops early! Warning: You're going to see a lot of WARNING print outputs from tf. I guess this tutorial is a bit out of date. But that's not what we care abot here, we just want that early stopping! The important output to notice is

```python
INFO:tensorflow:Validation (step 22556): accuracy = 0.966667, global_step = 22535, loss = 0.2767
INFO:tensorflow:Stopping. Best step: 22356 with loss = 0.2758353650569916.
```


```python
validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
                      "precision": tf.contrib.metrics.streaming_precision,
                      "recall": tf.contrib.metrics.streaming_recall}

validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50,
    #metrics=validation_metrics,
    early_stopping_metric='loss',
    early_stopping_metric_minimize=True,
    early_stopping_rounds=200)

tf.logging.set_verbosity(tf.logging.ERROR)
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
               monitors=[validation_monitor])
```




    Estimator(params={'dropout': None, 'hidden_units': [10, 20, 10], 'weight_column_name': None, 'feature_columns': [_RealValuedColumn(column_name='', dimension=4, default_value=None, dtype=tf.float32, normalizer=None)], 'optimizer': 'Adagrad', 'n_classes': 3, 'activation_fn': <function relu at 0x7f8568caa598>, 'num_ps_replicas': 0, 'gradient_clip_norm': None, 'enable_centered_bias': True})


