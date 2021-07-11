#!/bin/bash
set -e
mkdir inceptionv3
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz -q --show-progress --no-clobber
tar -xvf inception_v3_2016_08_28.tar.gz
mv inception_v3.ckpt inceptionv3
rm inception_v3_2016_08_28.tar.gz