#!/bin/bash

echo "Creating directories for data and model checkpoints..."
mkdir data
mkdir ./data/train_set
mkdir ./data/test_set
mkdir checkpoints

echo "Downloading bitbrains dataset..."
dataset_url="http://gwa.ewi.tudelft.nl/fileadmin/pds/trace-archives/\
grid-workloads-archive/datasets/gwa-t-12/fastStorage.zip"
wget $dataset_url -P ./data

echo "Extracting the dataset..."
unzip ./data/fastStorage.zip -d ./data

echo "Setting up paths for config file"
data_path=$(readlink -f ./data/fastStorage/2013-8)
train_path=$(readlink -f ./data/train_set)
test_path=$(readlink -f ./data/test_set)
checkpoint_path=$(readlink -f ./checkpoints)

echo "Generating new config file"
echo "---" > tmp_config.yaml
{
    echo "bitbrains_data_path: $data_path"
    echo "train_data_path: $train_path"
    echo "test_data_path: $test_path"
    echo "checkpoint_path: $checkpoint_path"
    echo "history_size: 90"
    echo "forecast_length: 6"
    echo "train: true"
    echo "plot: true"
    echo "epochs: 300"
    echo "machine_id: 253"
} >> tmp_config.yaml

echo "Overwriting old config file"
mv tmp_config.yaml config.yaml

echo "Creating train and test sets"
python3 generate_datasets.py