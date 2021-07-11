#!/bin/bash
set -e
mkdir resnet_v1_50
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz -q --show-progress --no-clobber
tar -xvf resnet_v1_50_2016_08_28.tar.gz
mv resnet_v1_50.ckpt resnet_v1_50
rm resnet_v1_50_2016_08_28.tar.gz