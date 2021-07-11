#!/bin/bash
set -e
mkdir inceptionv2
wget http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz -q --show-progress --no-clobber
tar -xvf inception_v2_2016_08_28.tar.gz
mv inception_v2.ckpt inceptionv2
rm inception_v2_2016_08_28.tar.gz