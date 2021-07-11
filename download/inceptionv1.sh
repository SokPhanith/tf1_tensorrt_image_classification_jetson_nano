#!/bin/bash
set -e
mkdir inceptionv1
wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz -q --show-progress --no-clobber
tar -xvf inception_v1_2016_08_28.tar.gz
mv inception_v1.ckpt inceptionv1
rm inception_v1_2016_08_28.tar.gz