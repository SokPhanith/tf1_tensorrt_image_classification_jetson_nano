#!/bin/bash
set -e
mkdir inception_resnetv2
wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz -q --show-progress --no-clobber
tar -xvf inception_resnet_v2_2016_08_30.tar.gz
mv inception_resnet_v2_2016_08_30.ckpt inception_resnetv2
rm inception_resnet_v2_2016_08_30.tar.gz
