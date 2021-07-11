#!/bin/sh
# Install TensorFlow models repository
echo "Install TensorFlow1 slim models repository"
url="https://github.com/tensorflow/models"
tf_models_dir="TF-models"
if [ ! -d "$tf_models_dir" ] ; then
	git clone $url $tf_models_dir
	cd "$tf_models_dir"/research
	git checkout 5f4d34fc
	cd slim
	sudo -H python3 setup.py install
fi

 
