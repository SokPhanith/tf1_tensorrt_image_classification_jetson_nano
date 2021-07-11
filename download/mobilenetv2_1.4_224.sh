#!/bin/bash
set -e
mkdir mobilenetv2_1.4_224
wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz -q --show-progress --no-clobber
tar -xvf mobilenet_v2_1.4_224.tgz
mv mobilenet_v2_1.4_224_frozen.pb mobilenetv2_1.4_224
rm mobilenet_v2_1.4_224.tgz
rm mobilenet_v2_1.4_224.tflite
rm mobilenet_v2_1.4_224.ckpt.meta
rm mobilenet_v2_1.4_224.ckpt.index
rm mobilenet_v2_1.4_224.ckpt.data-00000-of-00001
rm mobilenet_v2_1.4_224_info.txt
rm mobilenet_v2_1.4_224_eval.pbtxt

