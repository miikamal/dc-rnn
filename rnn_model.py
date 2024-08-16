import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from os import path
from utils import yaml_utils, data_utils
from sklearn.metrics import mean_squared_error

# Load in configs from config.yaml which needs to be in same directory as this
# python file
config = yaml_utils.read_config(path.dirname(path.abspath(__file__)) + \
    "/config.yaml")


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    '''
    Splits dataset into data and future timesteps to be predicted (labels)
    This function has been taken from Tensorflow LSTM Tutorial

        Parameters:
            dataset (): Path to the config.yaml file

        Returns:
            config (dict): Dictionary containing all config information
    '''
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

def plot_train_history(history, title):
    """Creates plot of train / validation loss from training history object"""
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()


# Load in training data and get the min and max values for scaling
train_df = data_utils.read_data(config)
min_values = train_df.min()
max_values = train_df.max()

if config.get("train"):

    # Take 80 % of data as training and rest to validation
    TRAIN_SPLIT = math.floor(0.8 * len(train_df["cpu_usage"]))

    # Normalize whole dataset
    normalized_train_df = (train_df-min_values)/(max_values-min_values)

    # Because some datasets has static values of some feature there is NA's
    # produced on normalization. Tensorflow doesnt like NaNs so we need to
    # replace those by 0
    normalized_train_df.fillna(value=0, inplace=True)

    # Predict cpu usage from all metrics given on data set
    features = normalized_train_df[[
        "cpu_usage", "memory_usage", "disk_read",
        "disk_write", "net_receive", "net_transmit"
    ]]

    # Generate the datasets
    dataset = features.values
    real_cpu = train_df["cpu_usage"].values
    x_train, y_train = multivariate_data(dataset, real_cpu, 0, TRAIN_SPLIT,
        config.get("history_size"), config.get("forecast_length"), 1
    )
    x_val, y_val = multivariate_data(dataset, real_cpu, TRAIN_SPLIT, None,
        config.get("history_size"), config.get("forecast_length"), 1
    )

# Setup parameters for data generators
BUFFER_SIZE = 10000
BATCH_SIZE = 32

# Create data generators for training set:
if config.get("train"):
    # Setup paramateres for training process
    STEPS_PER_EPOCH = math.ceil(len(x_train) / BATCH_SIZE)
    VALIDATION_STEPS = math.ceil(len(x_val) / BATCH_SIZE)
    # steps_per_epoch = number of training samples / batch size
    # validation steps = number of validation samples / batch size

    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = (
        train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    )

    x_val = x_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(BATCH_SIZE).repeat()

# Create model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(
    filters=35, kernel_size=6, input_shape=(config.get("history_size"), 6)
))
# If you want to use LSTM instead of GRU change the recurrent layer here
model.add(tf.keras.layers.GRU(1024))
model.add(tf.keras.layers.Dense(config.get("forecast_length")))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00012),
    loss='mse'
)

print("\n Number of trainable parameters layer by layer:\n")
model.summary()

# Setup model saving
checkpoint_base_path = (
    f"{config.get('checkpoint_path')}/"
    f"{config.get('machine_id')}"
)
checkpoint_path = checkpoint_base_path + "/cp.ckpt"
checkpoint_dir = path.dirname(checkpoint_path)
history_path = checkpoint_base_path + "/history.pkl"

# Callback that saves the model's weights when validation loss is lowest
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, save_best_only=True,
    monitor="val_loss", mode="min", verbose=1
)

if not config.get("train"):
    # Load training history
    with open(history_path, "rb") as f:
        history = pickle.load(f)
else:
    # Train the NN
    multi_step_history = model.fit(
        train_data, epochs=config.get("epochs"),
        steps_per_epoch=STEPS_PER_EPOCH, validation_data=val_data,
        validation_steps=VALIDATION_STEPS, callbacks=[cp_callback]
    )

    # Save history
    history = multi_step_history.history
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

if config.get("plot"):
    plot_train_history(history, "Training and validation loss (MSE)")



# Forecast of the test set starts below this line
# -----------------------------------------------

# Load in test data in the same way as the training data was loaded previously
test_df = data_utils.read_data(config, test_set=True)
normalized_test_df = (test_df-min_values)/(max_values-min_values)
normalized_test_df.fillna(value=0, inplace=True)
features = normalized_test_df[[
    "cpu_usage", "memory_usage", "disk_read",
    "disk_write", "net_receive", "net_transmit"
    ]]
dataset = features.values
real_cpu = test_df["cpu_usage"].values
x_test, y_test = multivariate_data(dataset, real_cpu, 0, None,
    config.get("history_size"), config.get("forecast_length"), 1
)

# Load pretrained weights which provided the lowest loss on validation set
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)

# Forecast
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
forecast = model.predict(x_test)

# Take every 6th (forecast length) prediction so same time point is not
# forecasted twice
forecast_array = []
for frcst in forecast[0::config.get("forecast_length")]:
    forecast_array.extend(frcst)
real_cpu = []
for y in y_test[0::config.get("forecast_length")]:
    real_cpu.extend(y)

# Clip out predictions to [0, 105] (max_value in training set for cpu was 104.4)
forecast_array = np.clip(forecast_array, 0, 105)

# Count RMSE in whole test set
rmse = mean_squared_error(real_cpu, forecast_array, squared=False)
rmse = round(rmse, 2)
print(f"\nRMSE in test set: {rmse}")
# There could be differences on RMSE of demo and paper.
# This depends on number of epochs trained, and model initial weights, biases

# Plot the forecast on test set
if config.get("plot"):
    plt.plot(
        range(len(forecast_array)), forecast_array, alpha=0.4, label="forecast"
    )
    plt.plot(
        range(len(real_cpu)), real_cpu, alpha=0.4,
        label ="true cpu usage"
    )
    plt.legend(loc="upper left")
    plt.title((
        f"GRU forecast for machine {config.get('machine_id')} "
        f"cpu_usage (RMSE: {rmse})"
    ))
    plt.show()