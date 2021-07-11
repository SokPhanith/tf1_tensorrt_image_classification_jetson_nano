#!/bin/bash
echo "remove old python file: imagenet.py"
rm TF-models/research/slim/datasets/imagenet.py
echo "copy new edit python file: imagenet.py."
cp scripts/imagenet.py TF-models/research/slim/datasets/
