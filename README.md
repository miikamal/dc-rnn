# Demo for ARIMA and GRU model training

This repository demonstrates the methods explained in the paper "Efficient Data Center Resource Usage Forecasting with Convolutional Recurrent Neural Networks" published in **TBA**.

## Description

This demo shows how the datasets and models were created. The training and forecasting process of ARIMA and GRU models are demonstrated as well.

## Getting Started

### Dependencies

* Python 3.8

* Python packages needed:
  * Pandas
  * numPy
  * scikit-learn
  * tqdm
  * TensorFlow
  * pmdarima
  * matplotlib
  * PyYAML
  ```
  pip install scikit-learn tqdm tensorflow pmdarima matplotlib PyYAML
  ```

* For automatical init of the repository (only available in Linux):
  * unzip
    * Ubuntu or Debian
    ```
    sudo apt install unzip
    ```
    * CentOS or Fedora
    ```
    sudo yum install unzip
    ```

### Installing

* Clone the repository 
  ```
  git clone https://github.com/miikamal/dc-rnn
  ```

* Automatical installation in Linux
  * cd into repository folder and run init shell script
    ```
    ./init.sh
    ```

* Manual installation
  1. Download the [GWA-T-12 fastStorage trace](http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains)
  2. Extract the dataset
  3. Set up paths in config.yaml
      * bitbrains_data_path: Path to folder having extracted csv files from step 2.
      * train_data_path: Path to folder where the train set should be saved
      * test_data_path: Path to folder where the test set should be saved
      * checkpoint_path: Path to folder where the models should be saved
  4. Run generate_datasets.py
      ```
      python generate_datasets.py
      ```

### Running the models

* config.yaml is the configuration file for running the demo. In addition to path parameters it has following options:
  * history_size (int): How many timepoints is fed into RNN model. Default is 90 which is same as in the original paper
  * forecast_length (int): How many timepoints in the future we forecast. Default is 6 which is same as in the original paper
  * train (boolean): Wether to train new model or forecast with pretrained model. However model needs to trained before it can be used to forecast only. Default is true
  * plot (boolean): Wether to show plots or not. This useful to set to false if training in remote machine. Default is true
  * epochs (int): How many epochs should the model be trained on. Default is 300
  * machine_id (int): Which machine should be forecasted, one of 220, 242, 253, 269, 283

* After configuring the config.yaml as explained the models can be run with following python scripts
  * GRU: rnn_model.py
    ```
    python rnn_model.py
    ```
  * ARIMA: arima_model.py
    ```
    python arima_model.py
    ```
