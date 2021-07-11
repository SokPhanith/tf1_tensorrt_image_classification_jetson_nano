#!/bin/bash
set -e
mkdir nasnet_mobile
wget https://storage.googleapis.com/download.tensorflow.org/models/nasnet-a_mobile_04_10_2017.tar.gz -q --show-progress --no-clobber
tar -xvf nasnet-a_mobile_04_10_2017.tar.gz
mv model.ckpt.index nasnet_mobile
mv model.ckpt.data-00000-of-00001 nasnet_mobile
rm nasnet-a_mobile_04_10_2017.tar.gz



