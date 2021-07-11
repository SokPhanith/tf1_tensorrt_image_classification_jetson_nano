#!/bin/bash
set -e
mkdir vgg_19
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz -q --show-progress --no-clobber
tar -xvf vgg_19_2016_08_28.tar.gz
mv vgg_19.ckpt vgg_19
rm vgg_19_2016_08_28.tar.gz
