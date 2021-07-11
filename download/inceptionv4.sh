#!/bin/bash
set -e
mkdir inceptionv4
wget http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz -q --show-progress --no-clobber
tar -xvf inception_v4_2016_09_09.tar.gz
mv inception_v4.ckpt inceptionv4
rm inception_v4_2016_09_09.tar.gz