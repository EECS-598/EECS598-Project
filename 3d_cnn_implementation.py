import tensorflow as tf
import numpy as np
import numpy as np
import nibabel as nib
import glob
import os

epochs = 1
learning_rate = 0.001
batch_size = 8
n_classes = 2
input_dimension = id = [256,256,88]
x = tf.placeholder("float", [None,id[0],id[1],id[2]])
y = tf.placeholder("float", [None, n_classes])
class_map = {'HC':0, 'PD':1}
curr_dir = os.getcwd()
directory = os.path.join(curr_dir,'PPMI')
# train_filenames = glob.glob(os.path.join(directory,'Train','**','*.nii.gz'))
# val_filenames = glob.glob(os.path.join(directory, 'Validation','**','*.nii.gz'))
# print('TOTAL TRAINING FILES ARE:  ', len(train_filenames))

train_tfrecords_path = os.path.join(directory,'Train','train_tfrecords-0.tfrecords')
model_dir = os.path.join(curr_dir,'Model_Checkpoint')

def input_model_fcn(features,labels,mode):
    conv1 = tf.layers.conv3d(inputs=features,filters=32,
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

    pool2_flat = tf.layers.flatten(pool2)
    dense1 = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1,units=128,activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense2, rate=0.4, training = mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=2)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

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

# def read_nii_image(filename,class_map):
#     # filename = tf.cast(filename,tf.string)
#     print(filename)
#     print('######################################')
#     a = tf.print(filename)
#     print(a)
#     print('######################################')
#     t1 = nib.load(filename)
#     t1_array = nib_t1.get_data()
#     t1 = t1[..., np.newaxis]
#     base_dir = os.path.dirname(filename)
#     class_name = os.path.basename(base_dir)
#     label = class_map[class_name]
#     return ({'features':image},label)
#
# def train_input_fcn(filenames,class_map,batch_size=100,shuffle=True):
#     print('************************')
#     print("Length of filenames ",len(filenames))
#     print('************************')
#     dataset  = tf.data.Dataset.from_tensor_slices(filenames)
#     print(dataset)
#     dataset = dataset.map(lambda x:read_nii_image(x,class_map)).shuffle(shuffle).batch(batch_size)
#     print('After:   ')
#     print(dataset)
#     ondisk_iterator = dataset.make_one_shot_iterator()
#     return ondisk_iterator.get_next()
def _parse_(serialized_example):
    feature = {'image_raw':tf.FixedLenFeature([],tf.string),
                'label':tf.FixedLenFeature([],tf.int64)}
    example = tf.parse_single_example(serialized_example,feature)
    image = tf.decode_raw(example['image_raw'],tf.float32) #remember to parse in int64. float will raise error
    image = tf.reshape(image,[256,256,88,1])
    label = tf.cast(example['label'],tf.int64)
    label = tf.one_hot(label,depth=2)
    return image, label
    # return ({'features':image},label)

def train_input_fcn(batch_size=8):
    tfrecord_dataset = tf.data.TFRecordDataset(train_tfrecords_path)
    tfrecord_dataset = tfrecord_dataset.map(lambda   x:_parse_(x)).shuffle(True).batch(batch_size)
    tfrecord_iterator = tfrecord_dataset.make_one_shot_iterator()

    return tfrecord_iterator.get_next()




def eval_input_fn(filenames,batch_size=100,shuffle=True):
    dataset  = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(lambda x:read_nii_image(x)).shuffle(shuffle).batch(batch_size)
    ondisk_iterator = dataset.make_one_shot_iterator()
    return ondisk_iterator.get_next()

parkinson_classifier = tf.estimator.Estimator(
    model_fn=input_model_fcn,model_dir=model_dir,
    # config = tf.estimator.RunConfig(
    #     save_checkpoints_steps = 2000,
    #     save_summary_steps = 100,
    #     keep_checkpoint_max=5)
    )
# parkinson_classifier.train(input_fn=lambda: train_input_fcn(train_filenames,class_map,batch_size=batch_size),steps=epochs)
parkinson_classifier.train(input_fn=lambda: train_input_fcn(batch_size=batch_size),steps=epochs)

print('Training done')
# eval_results = parkinson_classifier.evaluate(input_fn=eval_input_fn)
# print(eval_results)
