#!/bin/bash
set -e
mkdir resnet_v2_152
wget http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz -q --show-progress --no-clobber
tar -xvf resnet_v2_152_2017_04_14.tar.gz
mv resnet_v2_152.ckpt resnet_v2_152
rm resnet_v2_152_2017_04_14.tar.gz
rm eval.graph
rm train.graph