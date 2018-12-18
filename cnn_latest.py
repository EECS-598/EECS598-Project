import tensorflow as tf
import numpy as np
import numpy as np
# import nibabel as nib
import glob
import os

epochs = 1
learning_rate = 0.01
batch_size = 2

curr_dir = os.getcwd()
directory = os.path.join(curr_dir)

# train_tfrecords_path = os.path.join(directory,'PPMI','Train','train_tfrecords-1.tfrecords')

train_tfrecords_path = glob.glob('PPMI/Train/*')

test_tfrecords_path = glob.glob('PPMI/Test/*')

tf.logging.set_verbosity(tf.logging.INFO)

def input_model_fcn(features,labels,mode):

    conv1 = tf.layers.conv3d(inputs=features,filters=32,
                            kernel_size=[3,3,3],padding="same",
                            activation=tf.nn.relu)


#     conv2 = tf.layers.conv3d(inputs=conv1,filters=32,
#                             kernel_size=[3,3,3],padding="same",
#                             activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2,2,2], strides=2)

    conv3 = tf.layers.conv3d(inputs=pool1,filters=64,
                            kernel_size=[3,3,3],padding="same",
                            activation=tf.nn.relu)
#     conv4 = tf.layers.conv3d(inputs=conv3,filters=64,
#                             kernel_size=[3,3,3],padding="same",
#                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[4,4,4], strides=2)

    conv5 = tf.layers.conv3d(inputs=pool2,filters=128,
                            kernel_size=[3,3,3],padding="same",
                            activation=tf.nn.relu)
#     conv6 = tf.layers.conv3d(inputs=conv5,filters=128
#                             ,kernel_size=[3,3,3],padding="same",
#                             activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling3d(inputs=conv5, pool_size=[4,4,4], strides=2)

    pool3_flat = tf.layers.flatten(pool3)
    dense1 = tf.layers.dense(inputs=pool3_flat, units=512, activation=tf.nn.relu)


    dense2 = tf.layers.dense(inputs=pool3_flat,units=128,activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense2, rate=0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=tf.argmax(input=logits, axis=1))

  # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    logging_hook = tf.train.LoggingTensorHook({"loss":loss},every_n_iter=1)

      # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,training_hooks=[logging_hook])

      # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = { "accuracy": tf.metrics.accuracy(labels=tf.argmax(labels), predictions=tf.argmax(input=logits, axis=1))}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _parse_(serialized_example):
    feature = {'image':tf.FixedLenFeature([],tf.string),
                'label':tf.FixedLenFeature([],tf.int64)}
    example = tf.parse_single_example(serialized_example,feature)
    image = tf.decode_raw(example['image'],tf.float32) #remember to parse in int64. float will raise error
    print(image.get_shape())
    image = tf.reshape(image,[128,128,48,1])
    label = tf.cast(example['label'],tf.int64)
    label = tf.one_hot(label,depth=2)
    return image, label

# def train_input_fcn(batch_size=batch_size):
#
#     tfrecord_dataset = tf.data.TFRecordDataset([train_tfrecords_path])
#
#     tfrecord_dataset = tfrecord_dataset.map(lambda x:_parse_(x),num_parallel_calls=num_parallel_calls)
#
#     tfrecord_dataset = tfrecord_dataset.shuffle(buffer_size = 10000)
#     tfrecord_dataset = tfrecord_dataset.batch(batch_size)
#
#     tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()
#
#     return tfrecord_iterator.get_next()

def input_fn(is_training, filenames, batch_size, num_epochs=1, num_parallel_calls=1):
    print(filenames)
    dataset = tf.data.TFRecordDataset(filenames)

    if is_training:
        dataset = dataset.shuffle(buffer_size=1500)

    dataset = dataset.map(lambda value: _parse_(value),num_parallel_calls=num_parallel_calls)

    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()

def train_input_fn(file_path):
    return input_fn(True, file_path, batch_size, 5, 5)


def eval_input_fn(file_path):
    return input_fn(False, file_path, 2, 1, 1)


parkinson_classifier = tf.estimator.Estimator(
    model_fn=input_model_fcn, model_dir="model"
    #,
    #config = tf.estimator.RunConfig(
    #    save_checkpoints_steps = 0,
    #    save_summary_steps = 0,
    #    keep_checkpoint_max=0)
)


# parkinson_classifier.train(input_fn=lambda: train_input_fcn(train_filenames,class_map,batch_size=batch_size),steps=epochs)
parkinson_classifier.train(input_fn=lambda: train_input_fn(train_tfrecords_path))

print('Training done')
eval_results = parkinson_classifier.evaluate(input_fn= lambda: eval_input_fn(test_tfrecords_path))

print(eval_results)
