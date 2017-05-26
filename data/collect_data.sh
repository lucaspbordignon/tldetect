#!/bin/sh
#
# This script intend to be just for manual download of the dataset. The main
# jupyter-notebook downloads the dataset automatically.

# Downloading the full dataset (3.3GB)
echo "Downloading the dataset from Autti:"
wget https://s3.amazonaws.com/udacity-sdc/annotations/object-dataset.tar.gz
tar -xvf object-dataset.tar.gz
