#!/bin/bash
set -e
mkdir pnasnet_mobile
wget https://storage.googleapis.com/download.tensorflow.org/models/pnasnet-5_mobile_2017_12_13.tar.gz -q --show-progress --no-clobber
tar -xvf pnasnet-5_mobile_2017_12_13.tar.gz
mv model.ckpt.index pnasnet_mobile
mv model.ckpt.data-00000-of-00001 pnasnet_mobile
rm pnasnet-5_mobile_2017_12_13.tar.gz



