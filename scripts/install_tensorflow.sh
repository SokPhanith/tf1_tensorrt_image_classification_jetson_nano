#!/bin/bash
# Reference Site
# https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

echo "update and upgrade"
sudo apt-get update
sudo apt-get upgrade

# Install HDF5
echo "Install HDF5"
sudo apt-get -y install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
# Install pip
echo "Install pip"
sudo apt-get -y install python3-pip
sudo pip3 install -U pip testresources setuptools==49.6.0

# Install required packages
echo "Install required packages"
sudo apt-get -y install zlib1g-dev zip libjpeg8-dev libhdf5-dev
sudo pip3 install -U numpy==1.18.5 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.3.3 futures protobuf pybind11

# Install TensorFlow
echo "Install TensorFlow"
sudo pip3 install --no-cache-dir --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 "tensorflow<2"
