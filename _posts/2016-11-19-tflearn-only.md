---
title: TFLearn
layout: post
---
# Examples::Extending Tensorflow::Trainer


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


## Define the Architecture (Basic Tensorflow)


```python
# Because I don't feel like retyping stuff.
def tfp(shape):
    return tf.placeholder("float", shape)
def tfrn(shape, name):
    return tf.Variable(tf.random_normal(shape), name=name)

# Define the inputs/outputs/weights as usual.
X, Y       = tfp([None, 784]), tfp([None, 10])
W1, W2, W3 = tfrn([784, 256], 'W1'), tfrn([256, 256], 'W2'), tfrn([256, 10], 'W3')
b1, b2, b3 = tfrn([256], 'b1'), tfrn([256], 'b2'), tfrn([10], 'b3')

# Multilayer perceptron.
def dnn(x):
    x = tf.tanh(tf.add(tf.matmul(x, W1), b1))
    x = tf.tanh(tf.add(tf.matmul(x, W2), b2))
    x = tf.add(tf.matmul(x, W3), b3)
    return x
net = dnn(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
accuracy = tf.reduce_mean(tf.cast( 
        tf.equal( tf.argmax(net, 1), tf.argmax(Y, 1) ), tf.float32), 
        name='acc')
```

## Using  a TFLearn Trainer


```python
trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, metric=accuracy, batch_size=128)
trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=1)
```


```python
trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={X: testX, Y: testY}, 
           n_epoch=2, show_metric=True)
```

    Training Step: 860  | total loss: [1m[32m1.73376[0m[0m
    | Optimizer | epoch: 002 | loss: 1.73376 - acc: 0.8053 | val_loss: 1.78279 - val_acc: 0.8015 -- iter: 55000/55000
    Training Step: 860  | total loss: [1m[32m1.73376[0m[0m
    | Optimizer | epoch: 002 | loss: 1.73376 - acc: 0.8053 | val_loss: 1.78279 - val_acc: 0.8015 -- iter: 55000/55000
    --


# Training Callbacks

One suggestion for early stopping with tflearn (made by owner of tflearn repository) is to define a custom callback that raises an exception when we want to stop training. I've written a small snippet below as an example.


```python
class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, acc_thresh):
        """
        Args:
            acc_thresh - if our accuracy > acc_thresh, terminate training.
        """
        self.acc_thresh = acc_thresh
        self.accs = []
    
    def on_epoch_end(self, training_state):
        """ """
        self.accs.append(training_state.global_acc)
        if training_state.val_acc is not None and training_state.val_acc < self.acc_thresh:
            raise StopIteration
```


```python
cb = EarlyStoppingCallback(acc_thresh=0.5)
trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={X: testX, Y: testY}, 
           n_epoch=3, show_metric=True, snapshot_epoch=False,
            callbacks=cb)
```

    Training Step: 3965  | total loss: [1m[32m0.33810[0m[0m
    | Optimizer | epoch: 010 | loss: 0.33810 - acc: 0.9455 -- iter: 55000/55000
    GOODBYE



    ---------------------------------------------------------------------------

    StopIteration                             Traceback (most recent call last)

    <ipython-input-24-9c383c6f5a8b> in <module>()
          2 trainer.fit({X: trainX, Y: trainY}, val_feed_dicts={X: testX, Y: testY}, 
          3            n_epoch=3, show_metric=True, snapshot_epoch=False,
    ----> 4             callbacks=cb)
    

    /usr/local/lib/python3.5/dist-packages/tflearn/helpers/trainer.py in fit(self, feed_dicts, n_epoch, val_feed_dicts, show_metric, snapshot_step, snapshot_epoch, shuffle_all, dprep_dict, daug_dict, excl_trainops, run_id, callbacks)
        315 
        316                     # Epoch end
    --> 317                     caller.on_epoch_end(self.training_state)
        318 
        319             finally:


    /usr/local/lib/python3.5/dist-packages/tflearn/callbacks.py in on_epoch_end(self, training_state)
         67     def on_epoch_end(self, training_state):
         68         for callback in self.callbacks:
    ---> 69             callback.on_epoch_end(training_state)
         70 
         71     def on_train_end(self, training_state):


    <ipython-input-23-d44cbdbc0814> in on_epoch_end(self, training_state)
         13         if True:
         14             print("GOODBYE")
    ---> 15             raise StopIteration
    

    StopIteration: 



```python
cb.accs
```




    [None]




```python

```
