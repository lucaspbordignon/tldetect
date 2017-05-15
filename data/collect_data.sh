#!/bin/sh

# Downloading the full dataset (3.3GB)
echo "Downloading the dataset from Autti:"
curl -O https://s3.amazonaws.com/udacity-sdc/annotations/object-dataset.tar.gz
tar -xvf object-dataset.tar.gz
