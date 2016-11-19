# tutorial: tf.contrib.learn

# TODO: What are these . . .
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Iris training data sets
#IRIS_TRAINING = "iris_training.csv"
#IRIS_TEST = "iris_test.csv"


# Load datasets.
#training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        #filename=IRIS_TRAINING,
        #target_dtype=np.int,
        #features_dtype=np.float)
#
#test_set     = tf.contrib.learn.datasets.base.load_csv_with_header(
        #filename=IRIS_TEST,
        #target_dtype=np.int,
        #features_dtype=np.float)
#

iris = tf.contrib.learn.datasets.base.load_iris()

# Specify that all features have real-value data.
# TODO: What does real-value mean?
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3-layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        model_dir="/tmp/iris_model")

# Fit model.
# Note: State of model is preserved in classifier.
# See 'logging and monitoring basics with tf.contrib.learn' if want to track model while it learns.
classifier.fit(x=iris.data, y=iris.target, steps=2000)

#accuracy_score = classifier.evaluate(x=)


