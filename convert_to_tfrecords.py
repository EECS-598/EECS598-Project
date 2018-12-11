import tensorflow as tf
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
import glob

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_image(filename):
    sitk_t1 = sitk.ReadImage(filename)
    t1 = sitk.GetArrayFromImage(sitk_t1)
    t1 = t1[..., np.newaxis]
    return t1

def convert_to_tfrecord(dataset_name, data_directory, class_map, segments=1):

    files1 = '*.nii.gz'
    files2 = '*.nii'
    path1 = os.path.join(data_directory,'**',files1)
    path2 = os.path.join(data_directory,'**',files2)
    filenames1 = glob.glob(path1, recursive=True)
    filenames2 = glob.glob(path2, recursive=True)
    filenames = filenames1 + filenames2
    classes = (os.path.basename(os.path.dirname(name)) for name in filenames)
    dataset = list(zip(filenames, classes))

    num_examples = len(filenames)

    samples_per_segment = num_examples // segments

    print(f"Have {samples_per_segment} per record file")

    for segment_index in range(segments):
        start_index = segment_index * samples_per_segment
        end_index = (segment_index + 1) * samples_per_segment

        sub_dataset = dataset[start_index:end_index]
        record_filename = os.path.join(data_directory, f"{dataset_name}-{segment_index}.tfrecords")

        with tf.python_io.TFRecordWriter(record_filename) as writer:
            print(f"Writing {record_filename}")

            for index, sample in enumerate(sub_dataset):
                file_path, label = sample
                image_raw = load_image(file_path)
                image_raw = image_raw.tostring()

                features = {
                    'label': _int64_feature(class_map[label]),
                    'image': _bytes_feature(image_raw)
                }
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())



def main():

    current_dir = os.getcwd()

    train_dir = os.path.join(current_dir,'PPMI','Train')
    # validation_dir = os.path.join(current_dir,'PPMI','Validation')
    test_dir = os.path.join(current_dir,'PPMI','Test')

    train_filename = 'train_tfrecords'
    # validation_filename = 'validation_tfrecords'
    test_filename = 'test_tfrecords'

    name_to_label = {'PD':1, 'HC': 0}
    label_to_name = {1:"Parkinson's Disease", 2: 'Healthy'}

    convert_to_tfrecord(train_filename,train_dir,name_to_label,segments=4)
    # convert_to_tfrecord(validation_filename,validation_dir,name_to_label)
    convert_to_tfrecord(test_filename,test_dir,name_to_label)


if __name__ == '__main__':
    main()
