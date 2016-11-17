# Notebooks for learning TensorFlow and and TFLearn

Here you'll find notebooks containing tutorials I've worked through/tweaked related to [Tensorflow](https://www.tensorflow.org/) and [TFLearn](http://tflearn.org/).

More recently, I've also been looking into implementing early stopping in tflearn. In general, early stopping is when one exits the training process for a machine learning model. This is desirable for many reasons, but I'll be focusing on the case of validation accuracy saturation. Rather than having a deep neural network continue to train long after it's validation accuracy has peaked, I'd like to terminate training soon after this peak is reached.
