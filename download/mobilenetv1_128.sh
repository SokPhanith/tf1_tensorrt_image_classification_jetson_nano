#!/bin/bash
set -e
mkdir mobilenetv1_128
wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz -q --show-progress --no-clobber
tar -xvf mobilenet_v1_0.25_128.tgz
mv mobilenet_v1_0.25_128_frozen.pb mobilenetv1_128
rm mobilenet_v1_0.25_128.tgz
rm mobilenet_v1_0.25_128.tflite
rm mobilenet_v1_0.25_128.ckpt.meta
rm mobilenet_v1_0.25_128.ckpt.index
rm mobilenet_v1_0.25_128.ckpt.data-00000-of-00001
rm mobilenet_v1_0.25_128_info.txt
rm mobilenet_v1_0.25_128_eval.pbtxt

