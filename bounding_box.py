'''
   Copyright 2017 Larry Chen

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''

import os
import sys
import argparse
import collections
import json
import re
import time

import numpy as np
from PIL import Image
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'Path to CLEVR image directory')
flags.DEFINE_string('scene_file', '', 'Path to CLEVR scene file')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


def generate_label_map():
  sizes = ['large', 'small']
  colors = ['gray', 'red', 'blue', 'green', 'brown', 'purple', 'cyan', 'yellow']
  materials = ['rubber', 'metal']
  shapes = ['cube', 'sphere', 'cylinder']

  names = [s + ' ' + c + ' ' + m + ' ' + sh for s in sizes for c in colors for m in materials for sh in shapes]

  with open(os.path.join(FLAGS.output_path, 'clevr_label_map.pbtxt'), 'w') as f:
    [f.write('item {\n  id: %d\n  name: \'%s\'\n}\n\n' %(i+1, name)) for i, name in enumerate(names)]
    f.close()

  return names


def extract_bounding_boxes(scene, names):
  objs = scene['objects']
  rotation = scene['directions']['right']

  num_boxes = len(objs)

  boxes = np.zeros((1, num_boxes, 4))

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []

  for i, obj in enumerate(objs):
    [x, y, z] = obj['pixel_coords']

    [x1, y1, z1] = obj['3d_coords']

    cos_theta, sin_theta, _ = rotation

    x1 = x1 * cos_theta + y1* sin_theta
    y1 = x1 * -sin_theta + y1 * cos_theta


    height_d = 6.9 * z1 * (15 - y1) / 2.0
    height_u = height_d
    width_l = height_d
    width_r = height_d

    if obj['shape'] == 'cylinder':
      d = 9.4 + y1
      h = 6.4
      s = z1

      height_u *= (s*(h/d + 1)) / ((s*(h/d + 1)) - (s*(h-s)/d))
      height_d = height_u * (h-s+d)/ (h + s + d)

      width_l *= 11/(10 + y1)
      width_r = width_l

    if obj['shape'] == 'cube':
      height_u *= 1.3 * 10 / (10 + y1)
      height_d = height_u
      width_l = height_u
      width_r = height_u
    
    obj_name = obj['size'] + ' ' + obj['color'] + ' ' + obj['material'] + ' ' + obj['shape']
    classes_text.append(obj_name.encode('utf8'))
    classes.append(names.index(obj_name) + 1)
    ymin.append((y - height_d)/320.0)
    ymax.append((y + height_u)/320.0)
    xmin.append((x - width_l)/480.0)
    xmax.append((x + width_r)/480.0)

  return xmin, ymin, xmax, ymax, classes, classes_text


def file_to_example_dict(scene_file, names):
  with open(scene_file) as sf:
    scene_data = json.load(sf)
    scenes = scene_data['scenes']

  examples = []

  for scene in scenes:
    xmins, ymins, xmaxs, ymaxs, classes, classes_text = extract_bounding_boxes(scene, names)

    example = {
      'xmins': xmins,
      'ymins': ymins,
      'xmaxs': xmaxs,
      'ymaxs': ymaxs,
      'classes': classes,
      'classes_text': classes_text,
      'filename': scene['image_filename']
    }
    examples.append(example)

  return examples


def create_tf_example(example, image_dir):
  # TODO(user): Populate the following variables from your example.
  img_path = os.path.join(image_dir, example['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_png = fid.read()

  height = 320
  width = 480
  filename = example['filename']
  image_format = 'png'

  xmins = example['xmins']
  xmaxs = example['xmaxs']
  ymins = example['ymins']
  ymaxs = example['ymaxs']

  classes_text = example['classes_text']
  classes = example['classes']

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
      'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
      'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
      'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
      'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_png])),
      'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format.encode('utf8')])),
      'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
      'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
      'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
      'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
      'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
      'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes))
  }))

  return tf_example

def write_tf_examples(names):
  examples = file_to_example_dict(FLAGS.scene_file, names)

  writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path, 'clevr_object_detect.tfrecord'))

  for i, example in enumerate(examples):
    tf_example = create_tf_example(example, FLAGS.image_dir)
    if i % 100 == 0:
      print('\rOn image %d of %d' %(i, len(examples)), end='')
    writer.write(tf_example.SerializeToString())

  writer.close()

def main(_):
  names = generate_label_map()
  write_tf_examples(names)


if __name__ == '__main__':
  tf.app.run()