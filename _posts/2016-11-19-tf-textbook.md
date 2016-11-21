---
title: TensorFlow Textbook Tutorials
layout: post
---
# Using Tensorboard


```python
import tensorflow as tf
a = tf.constant(10, name="a")
b = tf.constant(90, name="b")
y = tf.Variable(a + 2 * b, name="y")

model = tf.initialize_all_variables()
with tf.Session() as session:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter\
    ("/tmp/tensorflowlogs", session.graph)
    session.run(model)
    print(session.run(y))
    
# Open terminal and run command:
# tensorboard --logdir=/tmp/tensorflowlogs
```

    190


# MNIST Convolutional NN


```python
import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
```

    Extracting /tmp/data/train-images-idx3-ubyte.gz
    Extracting /tmp/data/train-labels-idx1-ubyte.gz
    Extracting /tmp/data/t10k-images-idx3-ubyte.gz
    Extracting /tmp/data/t10k-labels-idx1-ubyte.gz



```python
n_input, n_classes = 784, 10

# Hyperparameters
learning_rate  = 1e-3
training_iters = 1e5
batch_size     = 128
display_step   = 10
dropout        = 0.75 # dropout probability
keep_prob = tf.placeholder(tf.float32) # (for dropout)

x = tf.placeholder(tf.float32, [None, n_input])
_X = tf.reshape(x, shape=[-1, 28, 28, 1])  # Assuming -1 will be the number of samples?
y = tf.placeholder(tf.float32, [None, n_classes]) # output probabilities
```


```python
def conv2d(img, w, b):
    """ 
    Args:
        img --  input tensor of shape [batchsize, in_height, in_width, in_channels]
                where channels may be, e.g. 3 for RGB color
        w   --  filter with shape [f_height, f_width, in_channels, n_feat_maps]
        b   --  bias for each feature map (number of biases = depth of the conv layer)
    """
    return tf.nn.relu(tf.nn.bias_add(\
            tf.nn.conv2d(img, w, strides=[1,1,1,1], padding='SAME'), b))

def max_pool(img, k=2):
    """
    Args:
        img -- output of a conv layer
        k   -- window size and stride (small)
    """
    return tf.nn.max_pool(img, 
                         ksize=[1, k, k, 1], 
                         strides=[1, k, k, 1], 
                         padding='SAME')
```


```python
# __________ Weights and biases for all layers __________

# 5x5 conv, 1 input, 32 outputs
wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32])) 
bc1 = tf.Variable(tf.random_normal([32]))

# 5x5 conv, 32 inputs, 64 outputs
wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64])) 
bc2 = tf.Variable(tf.random_normal([64]))

# FC, 7*7*64 inputs, 1024 outputs
wd1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
bd1 = tf.Variable(tf.random_normal([1024]))

# Output layer. 1024 inputs, 10 outputs.
wout = tf.Variable(tf.random_normal([1024, n_classes]))
bout = tf.Variable(tf.random_normal([n_classes]))
```


```python
# __________ The layers __________

# [In] --> Conv --> Pool --> Dropout
conv1 = conv2d(_X, wc1, bc1)
conv1 = max_pool(conv1, k=2)
conv1 = tf.nn.dropout(conv1, keep_prob)

# --> Conv --> Pool --> Dropout
conv2 = conv2d(conv1, wc2, bc2)
conv2 = max_pool(conv2, k=2)
conv2 = tf.nn.dropout(conv2, keep_prob)

# --> Fully-Connected[ReLu] --> Dropout
# (reshape conv2 out essentially by flattening all maps into single list)
dense1 = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
dense1 = tf.nn.relu( tf.add( tf.matmul( dense1, wd1 ), bd1 ) )
dense1 = tf.nn.dropout(dense1, keep_prob)

# Output prediction.
pred = tf.add(tf.matmul(dense1, wout), bout)
```

## Cost and Optimizing

$$
\text{cost} = \frac{1}{n} \sum_{i = 1}^{n_{out}} y_i \log\bigg( \frac{e^{z_i}}{\sum_k e^{z_k}}\bigg)
$$


```python
# ______________ Training _______
cost      = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
```


```python
# _______ Evaluation _____
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy     = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```


```python
# _________ BLAST OFF _____________
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter", step * batch_size, 
                 ", Minibatch Loss={:.6f}".format(loss),
                 ", Training Accuracy={:.5f}".format(acc))
        
        step += 1
    print("Optimization finished. Am robot.")





















```

    ('Iter', 1280, ', Minibatch Loss=25021.994141', ', Training Accuracy=0.26562')
    ('Iter', 2560, ', Minibatch Loss=20956.230469', ', Training Accuracy=0.41406')
    ('Iter', 3840, ', Minibatch Loss=10467.468750', ', Training Accuracy=0.54688')
    ('Iter', 5120, ', Minibatch Loss=6931.669434', ', Training Accuracy=0.64844')
    ('Iter', 6400, ', Minibatch Loss=11381.146484', ', Training Accuracy=0.58594')
    ('Iter', 7680, ', Minibatch Loss=6931.756836', ', Training Accuracy=0.67188')
    ('Iter', 8960, ', Minibatch Loss=6043.289062', ', Training Accuracy=0.70312')
    ('Iter', 10240, ', Minibatch Loss=2950.967041', ', Training Accuracy=0.78906')
    ('Iter', 11520, ', Minibatch Loss=4387.661133', ', Training Accuracy=0.79688')
    ('Iter', 12800, ', Minibatch Loss=4279.759277', ', Training Accuracy=0.78125')
    ('Iter', 14080, ', Minibatch Loss=2511.234863', ', Training Accuracy=0.84375')
    ('Iter', 15360, ', Minibatch Loss=3200.528809', ', Training Accuracy=0.79688')
    ('Iter', 16640, ', Minibatch Loss=2861.273438', ', Training Accuracy=0.82031')
    ('Iter', 17920, ', Minibatch Loss=2214.196289', ', Training Accuracy=0.88281')
    ('Iter', 19200, ', Minibatch Loss=989.559265', ', Training Accuracy=0.90625')
    ('Iter', 20480, ', Minibatch Loss=4211.814941', ', Training Accuracy=0.78906')
    ('Iter', 21760, ', Minibatch Loss=1644.427979', ', Training Accuracy=0.91406')
    ('Iter', 23040, ', Minibatch Loss=2109.490967', ', Training Accuracy=0.87500')
    ('Iter', 24320, ', Minibatch Loss=2386.041504', ', Training Accuracy=0.83594')
    ('Iter', 25600, ', Minibatch Loss=1501.948364', ', Training Accuracy=0.88281')
    ('Iter', 26880, ', Minibatch Loss=2240.972656', ', Training Accuracy=0.82812')
    ('Iter', 28160, ', Minibatch Loss=2119.425537', ', Training Accuracy=0.87500')
    ('Iter', 29440, ', Minibatch Loss=2242.839844', ', Training Accuracy=0.82812')
    ('Iter', 30720, ', Minibatch Loss=1093.348633', ', Training Accuracy=0.88281')
    ('Iter', 32000, ', Minibatch Loss=1532.251099', ', Training Accuracy=0.88281')
    ('Iter', 33280, ', Minibatch Loss=985.126221', ', Training Accuracy=0.88281')
    ('Iter', 34560, ', Minibatch Loss=1191.394165', ', Training Accuracy=0.90625')
    ('Iter', 35840, ', Minibatch Loss=2769.808105', ', Training Accuracy=0.82812')
    ('Iter', 37120, ', Minibatch Loss=451.285889', ', Training Accuracy=0.94531')
    ('Iter', 38400, ', Minibatch Loss=857.569580', ', Training Accuracy=0.89844')
    ('Iter', 39680, ', Minibatch Loss=2352.155762', ', Training Accuracy=0.88281')
    ('Iter', 40960, ', Minibatch Loss=1384.690674', ', Training Accuracy=0.90625')
    ('Iter', 42240, ', Minibatch Loss=828.415405', ', Training Accuracy=0.92188')
    ('Iter', 43520, ', Minibatch Loss=437.712341', ', Training Accuracy=0.95312')
    ('Iter', 44800, ', Minibatch Loss=584.637817', ', Training Accuracy=0.89844')
    ('Iter', 46080, ', Minibatch Loss=1383.199707', ', Training Accuracy=0.89062')
    ('Iter', 47360, ', Minibatch Loss=1923.911255', ', Training Accuracy=0.88281')
    ('Iter', 48640, ', Minibatch Loss=1327.275146', ', Training Accuracy=0.88281')
    ('Iter', 49920, ', Minibatch Loss=450.466156', ', Training Accuracy=0.90625')
    ('Iter', 51200, ', Minibatch Loss=461.589783', ', Training Accuracy=0.93750')
    ('Iter', 52480, ', Minibatch Loss=512.834595', ', Training Accuracy=0.95312')
    ('Iter', 53760, ', Minibatch Loss=1481.610840', ', Training Accuracy=0.85156')
    ('Iter', 55040, ', Minibatch Loss=1503.613281', ', Training Accuracy=0.90625')
    ('Iter', 56320, ', Minibatch Loss=663.131042', ', Training Accuracy=0.91406')
    ('Iter', 57600, ', Minibatch Loss=836.979126', ', Training Accuracy=0.94531')
    ('Iter', 58880, ', Minibatch Loss=1394.500244', ', Training Accuracy=0.92188')
    ('Iter', 60160, ', Minibatch Loss=1150.654297', ', Training Accuracy=0.89062')
    ('Iter', 61440, ', Minibatch Loss=884.085022', ', Training Accuracy=0.89844')
    ('Iter', 62720, ', Minibatch Loss=641.650208', ', Training Accuracy=0.93750')
    ('Iter', 64000, ', Minibatch Loss=612.565613', ', Training Accuracy=0.92188')
    ('Iter', 65280, ', Minibatch Loss=1026.186890', ', Training Accuracy=0.88281')
    ('Iter', 66560, ', Minibatch Loss=1012.022217', ', Training Accuracy=0.89844')
    ('Iter', 67840, ', Minibatch Loss=538.746582', ', Training Accuracy=0.92969')
    ('Iter', 69120, ', Minibatch Loss=2331.966064', ', Training Accuracy=0.85156')
    ('Iter', 70400, ', Minibatch Loss=611.249207', ', Training Accuracy=0.92969')
    ('Iter', 71680, ', Minibatch Loss=611.909607', ', Training Accuracy=0.94531')
    ('Iter', 72960, ', Minibatch Loss=1363.580566', ', Training Accuracy=0.88281')
    ('Iter', 74240, ', Minibatch Loss=996.121582', ', Training Accuracy=0.91406')
    ('Iter', 75520, ', Minibatch Loss=730.850952', ', Training Accuracy=0.92969')
    ('Iter', 76800, ', Minibatch Loss=781.747681', ', Training Accuracy=0.92969')
    ('Iter', 78080, ', Minibatch Loss=854.089539', ', Training Accuracy=0.93750')
    ('Iter', 79360, ', Minibatch Loss=1397.916870', ', Training Accuracy=0.88281')
    ('Iter', 80640, ', Minibatch Loss=1405.003418', ', Training Accuracy=0.88281')
    ('Iter', 81920, ', Minibatch Loss=806.627136', ', Training Accuracy=0.92188')
    ('Iter', 83200, ', Minibatch Loss=647.945007', ', Training Accuracy=0.93750')
    ('Iter', 84480, ', Minibatch Loss=1018.518982', ', Training Accuracy=0.93750')
    ('Iter', 85760, ', Minibatch Loss=1204.980469', ', Training Accuracy=0.89062')
    ('Iter', 87040, ', Minibatch Loss=743.574951', ', Training Accuracy=0.92188')
    ('Iter', 88320, ', Minibatch Loss=638.823486', ', Training Accuracy=0.95312')
    ('Iter', 89600, ', Minibatch Loss=549.751770', ', Training Accuracy=0.96094')
    ('Iter', 90880, ', Minibatch Loss=727.560242', ', Training Accuracy=0.91406')
    ('Iter', 92160, ', Minibatch Loss=624.963196', ', Training Accuracy=0.91406')
    ('Iter', 93440, ', Minibatch Loss=1152.272461', ', Training Accuracy=0.85938')
    ('Iter', 94720, ', Minibatch Loss=409.238037', ', Training Accuracy=0.95312')
    ('Iter', 96000, ', Minibatch Loss=444.576447', ', Training Accuracy=0.92969')
    ('Iter', 97280, ', Minibatch Loss=1209.410645', ', Training Accuracy=0.86719')
    ('Iter', 98560, ', Minibatch Loss=217.887985', ', Training Accuracy=0.93750')
    ('Iter', 99840, ', Minibatch Loss=469.807068', ', Training Accuracy=0.92969')
    Optimization finished. Am robot.



```python

```
