#!/bin/bash
set -e
mkdir mobilenetv1_160
wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz -q --show-progress --no-clobber
tar -xvf mobilenet_v1_0.5_160.tgz
mv mobilenet_v1_0.5_160_frozen.pb mobilenetv1_160
rm mobilenet_v1_0.5_160.tgz
rm mobilenet_v1_0.5_160.tflite
rm mobilenet_v1_0.5_160.ckpt.meta
rm mobilenet_v1_0.5_160.ckpt.index
rm mobilenet_v1_0.5_160.ckpt.data-00000-of-00001
rm mobilenet_v1_0.5_160_info.txt
rm mobilenet_v1_0.5_160_eval.pbtxt

