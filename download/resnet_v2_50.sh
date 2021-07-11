#!/bin/bash
set -e
mkdir resnet_v2_50
wget http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz -q --show-progress --no-clobber
tar -xvf resnet_v2_50_2017_04_14.tar.gz
mv resnet_v2_50.ckpt resnet_v2_50
rm resnet_v2_50_2017_04_14.tar.gz
rm eval.graph
rm train.graph