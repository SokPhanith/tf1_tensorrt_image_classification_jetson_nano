#!/bin/bash
set -e
mkdir vgg_16
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz -q --show-progress --no-clobber
tar -xvf vgg_16_2016_08_28.tar.gz
mv vgg_16.ckpt vgg_16
rm vgg_16_2016_08_28.tar.gz
