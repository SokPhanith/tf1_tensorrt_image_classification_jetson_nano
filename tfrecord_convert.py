import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import random
import sys
import tensorflow as tf
import argparse
def get_parser():
    parser = argparse.ArgumentParser(description="Convert image dataset to tfrecord dataset")
    parser.add_argument("--dataset_dir",default="images/board",help="path to dataset model.")
    parser.add_argument("--num_test",type=int,default=300,help="number image for test default[300].")
    parser.add_argument("--random_seed",type=int,default=0,help="random seed number default[0].")
    parser.add_argument("--num_shards",type=int,default=5,help="number of output file TFrecord default[5].")
    return parser
args = get_parser().parse_args()
_NUM_VALIDATION = args.num_test
_RANDOM_SEED = args.random_seed
_NUM_SHARDS = args.num_shards 
_DATASET_NAME = 'flowers'
dataset_dir = args.dataset_dir
class ImageReader(object):
  def __init__(self):
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]
  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image
def int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
def bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),}))
def _get_filenames_and_classes(dataset_dir):
  flower_root = dataset_dir
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)
  return photo_filenames, sorted(class_names)

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = _DATASET_NAME+'_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)
def write_label_file(labels_to_class_names, dataset_dir):
  labels_filename = os.path.join(dataset_dir, 'labels.txt')
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))
def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  assert split_name in ['train', 'validation']
  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
  with tf.Graph().as_default():
    image_reader = ImageReader()
    with tf.Session('') as sess:
      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(filenames), shard_id))
            sys.stdout.flush()
  
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = image_to_tfexample(image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()


photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
class_names_to_ids = dict(zip(class_names, range(len(class_names))))

random.seed(_RANDOM_SEED)
random.shuffle(photo_filenames)
training_filenames = photo_filenames[_NUM_VALIDATION:]
validation_filenames = photo_filenames[:_NUM_VALIDATION]

_convert_dataset('train', training_filenames, class_names_to_ids,dataset_dir)
_convert_dataset('validation', validation_filenames, class_names_to_ids,dataset_dir)

labels_to_class_names = dict(zip(range(len(class_names)), class_names))
write_label_file(labels_to_class_names, dataset_dir)

print('\nFinished converting the dataset!')
