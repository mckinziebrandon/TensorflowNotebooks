---
title: Misc. TensorFlow Tutorials
layout: post
---

```python
import numpy as np
import tensorflow as tf
```

# Hello World
* Based on [this tutorial](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/helloworld.ipynb "I'm watching you")


```python
hello = tf.constant('Hello, Tensorflow.') # Create a constant op, added as node to default graph.
sess = tf.Session()                       # Start tensorflow session.
print sess.run(hello)                     # Run graph.
```

# Basic Operations

* Based on [this tutorial](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/basic_operations.ipynb "Still watching")
* __Constants__: Directly perform arithmetic with tf.constants within sess.run().
* __Variables__: (i.e. tf.placeholder) need to provide feed_dict of values. 
* __Matrix Multiplication__: Here, define matrices as constants, and pass to tf.matmul.
    * No feed_dict necessary.


```python
# Actual numerical values used in the examples below.
_a, _b = 2, 3
_matrix1, _matrix2 = [[3., 3.]], [[2.], [2.]]

# __________ Example: tf.constant ____________
a = tf.constant(_a)
b = tf.constant(_b)
with tf.Session() as sess:
    print "a, b = ({0}, {1})".format(_a, _b)
    print "Addition with constants: a + b = %i " % sess.run(a + b)
    print "Multiplication with constants: a * b = %i " % sess.run(a * b)

# __________ Example: tf.placeholder ____________
a     = tf.placeholder(tf.int16)
b     = tf.placeholder(tf.int16)
add   = tf.add(a, b)
mult  = tf.mul(a, b)
with tf.Session() as sess:
    feed_dict = {a: _a, b: _b}
    print "
In [ ]:
Addition with constants: a + b = %i " % sess.run(add, feed_dict)
    print "Multiplication with constants: a * b = %i " % sess.run(mult, feed_dict)

# __________ Example: tf.matmul ____________
matrix1     = tf.constant(_matrix1)
matrix2     = tf.constant(_matrix2)
matrix_product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    print "Matrix multiply 1x2 * 2x1 matrices: {0}".format(sess.run(matrix_product))
```

# Nearest-Neighbors on MNIST

* Based on [this tutorial](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/nearest_neighbor.ipynb "Hi")


```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
```


```python
_Xtrain, _Ytrain = mnist.train.next_batch(500)
_Xtest, _Ytest   = mnist.test.next_batch(200)
d = _Xtrain.shape[1]

# Graph input.
Xtrain = tf.placeholder("float", [None, d]) # I think 'None' here means it can be whatever. 
Xtest = tf.placeholder("float", [d])

# L1 distance between Xtrain and Xtest (why?)
difference = tf.add(Xtrain, tf.neg(Xtest))
L1_dist    = tf.reduce_sum(tf.abs(difference), reduction_indices=1)

# Prediction : get nearest neighbor. 
pred = tf.arg_min(L1_dist, 0)

accuracy = 0.
init = tf.initialize_all_variables() # TODO: Forgot what this does...

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(_Xtest)):
        feed_dict = {Xtrain: _Xtrain, Xtest: _Xtest[i, :]}
        nearest_neighbor_index = sess.run(pred, feed_dict)
        
        label_train = np.argmax(_Ytrain[nearest_neighbor_index])
        label_test  = np.argmax(_Ytest[i])
        if label_train == label_test:
            accuracy += 1. / len(_Xtest)

    print "Accuracy:", accuracy
```

# TFLearn - Quick Start


```python
np.info(np.reshape)import numpy as np
import tflearn
import tensorflow as tf
from tflearn.datasets import titanic

data = titanic.download_dataset('titanic_dataset.csv')
                                        
from tflearn.data_utils import load_csv
data, labels = load_csv('titanic_dataset.csv', 
                       target_column=0,
                       categorical_labels=True,
                       n_classes=2)

# ___________ Data Preprocessing ___________

def preprocess(data, columns_to_ignore):
    
    # Sort by descending id and delete columns. 
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data] 
    
    for i in range(len(data)):
        # Encode male=0, female=1. 
        data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)

to_ignore = [1, 6]
data = preprocess(data, to_ignore)


# ___________ Build the DNN ___________
# Input --> FC --> FC --> SOFTMAX
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# __________ Training ____________
model = tflearn.DNN(net)
model

```


```python
"""
Simple Example to train logical operators
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tflearn

'''
Going further: Graph combination with multiple optimizers
Create a XOR operator using product of NAND and OR operators
'''
# Data
X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
Y_nand = [[1.], [1.], [1.], [0.]]
Y_or = [[0.], [1.], [1.], [1.]]

# Graph definition
with tf.Graph().as_default():
    
    # Building a network with 2 optimizers
    g = tflearn.input_data(shape=[None, 2])
    # Nand operator definition
    g_nand = tflearn.fully_connected(g, 32, activation='linear')
    g_nand = tflearn.fully_connected(g_nand, 32, activation='linear')
    g_nand = tflearn.fully_connected(g_nand, 1, activation='sigmoid')
    g_nand = tflearn.regression(g_nand, optimizer='sgd',
                                learning_rate=2.,
                                loss='binary_crossentropy')
    
    # Or operator definition
    g_or = tflearn.fully_connected(g, 32, activation='linear')
    g_or = tflearn.fully_connected(g_or, 32, activation='linear')
    g_or = tflearn.fully_connected(g_or, 1, activation='sigmoid')
    g_or = tflearn.regression(g_or, optimizer='sgd',
                              learning_rate=2.,
                              loss='binary_crossentropy')
    
    # XOR merging Nand and Or operators
    g_xor = tflearn.merge([g_nand, g_or], mode='elemwise_mul')

    # Training
    m = tflearn.DNN(g_xor)
    m.fit(X, [Y_nand, Y_or], n_epoch=400, snapshot_epoch=False)

    # Testing
    print("Testing XOR operator") 
    print("0 xor 0:", m.predict([[0., 0.]]))
    print("0 xor 1:", m.predict([[0., 1.]]))
    print("1 xor 0:", m.predict([[1., 0.]]))
    print("1 xor 1:", m.predict([[1., 1.]]))
```

    Training Step: 400  | total loss: [1m[32m0.81728[0m[0m
    | SGD_0 | epoch: 400 | loss: 0.40857 -- iter: 4/4
    | SGD_1 | epoch: 400 | loss: 0.40871 -- iter: 4/4
    Testing XOR operator
    0 xor 0: [[0.0005703496863134205]]
    0 xor 1: [[0.9982306957244873]]
    1 xor 0: [[0.9982070922851562]]
    1 xor 1: [[0.00094714475562796]]


# Early Stopping Investigatoin


```python
import csv
import tensorflow as tf
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline
ipd = pd.read_csv('iris.csv')
ipd.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
