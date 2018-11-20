import tensorflow as tf
from tensorflow import keras
import numpy as np

epochs = 500
learning_rate = 0.001
batch_size = 8
n_classes = 2
input_dimension = id = [80,100,108]
x = tf.placeholder("float", [None,id[0],id[1],id[2]])
y = tf.placeholder("float", [None, n_classes])

def input_model_fcn(input,labels,mode):
    conv1 = tf.layers.conv3d(inputs=input_layer,filters=32,
                            kernel_size=[3,3,3],padding="same",
                            activation=tf.nn.relu)
    conv2 = tf.layers.conv3d(inputs=conv1,filters=32,
                            kernel_size=[3,3,3],padding="same",
                            activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2,2,2], strides=2)

    conv3 = tf.layers.conv3d(inputs=pool1,filters=64,
                            kernel_size=[3,3,3],padding="same",
                            activation=tf.nn.relu)
    conv4 = tf.layers.conv3d(inputs=conv3,filters=64,
                            kernel_size=[3,3,3],padding="same",
                            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[4,4,4], strides=2)

    conv5 = tf.layers.conv3d(inputs=pool2,filters=128,
                            kernel_size=[3,3,3],padding="same",
                            activation=tf.nn.relu)
    conv6 = tf.layers.conv3d(inputs=conv5,filters=128
                            ,kernel_size=[3,3,3],padding="same",
                            activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling3d(inputs=conv6, pool_size=[4,4,4], strides=2)

    pool2_flat = tf.flatten(pool2)
    dense1 = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1,units=128,activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(labels=labels, logits=logits)

      # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

      # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def train_input_fcn(x={"x": train_data},y=train_labels,batch_size=100,num_epochs=None,shuffle=True):

    pass

def eval_input_fn(x={"x": eval_data},y=eval_labels,num_epochs=1,shuffle=False):
    pass

parkinson_classifier = tf.estimator.Estimator(
    model_fn=input_model_fcn)
parkinson_classifier.train(input_fn=train_input_fn,steps=epochs)

eval_results = parkinson_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)
