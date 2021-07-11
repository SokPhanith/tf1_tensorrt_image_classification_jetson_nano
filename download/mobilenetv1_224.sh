#!/bin/bash
set -e
mkdir mobilenetv1_224
wget http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz -q --show-progress --no-clobber
tar -xvf mobilenet_v1_1.0_224.tgz
mv mobilenet_v1_1.0_224_frozen.pb mobilenetv1_224
mv mobilenet_v1_1.0_224.ckpt.meta mobilenetv1_224
mv mobilenet_v1_1.0_224.ckpt.index mobilenetv1_224
mv mobilenet_v1_1.0_224.ckpt.data-00000-of-00001 mobilenetv1_224
rm mobilenet_v1_1.0_224.tgz
rm mobilenet_v1_1.0_224.tflite
rm mobilenet_v1_1.0_224_info.txt
rm mobilenet_v1_1.0_224_eval.pbtxt

