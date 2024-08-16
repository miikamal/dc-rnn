from utils import yaml_utils, data_utils
from os import path, makedirs
from tqdm import tqdm
from math import floor
from sklearn.metrics import mean_squared_error
import pmdarima as pm
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load in configs from config.yaml which needs to be in same directory as this
# python file
config = yaml_utils.read_config(path.dirname(path.abspath(__file__)) + \
    "/config.yaml")

# Path where the model is saved on disk
MODEL_PATH = (
    f"{config.get('checkpoint_path')}/{config.get('machine_id')}/"
    f"arima_model.pkl"
)

if config.get("train"):
    print(f"Starting to train arima model for machine {config.get('machine_id')}")

    # Load data
    train_df = data_utils.read_data(config)
    train_cpu_series = train_df["cpu_usage"].values

    print("Finding optimal hyperparameters for arima model...")
    # Use auto_arima to find optimal order of arima
    model = pm.auto_arima(train_cpu_series)
    print(f"\tAuto arima parameters: {model.order}")


    # Save the model to disk
    if not path.exists(path.dirname(MODEL_PATH)):
        makedirs(path.dirname(MODEL_PATH))
    with open(MODEL_PATH, 'wb+') as pkl:
                pickle.dump(model, pkl)
else:
    print(f"Loading pretrained model for machine {config.get('machine_id')}")
    with open(MODEL_PATH, 'rb') as pkl:
            model = pickle.load(pkl)


# Forecast with the model
print("Starting to forecast with the model...")
test_df = data_utils.read_data(config, test_set=True)
test_cpu_series = test_df["cpu_usage"].values

# Loop through whole test set at 30 mins intervals and do predictions for next
# 30 mins (=6 timesteps)
forecast_interval = config.get("forecast_length")
forecast_array = []
for i in tqdm(range(floor(len(test_cpu_series) / forecast_interval))):
    forecast = model.predict(n_periods=forecast_interval)
    forecast_array.extend(forecast)
    model.update(
        test_cpu_series[(i * forecast_interval) : ((i+1) * forecast_interval)],
        maxiter=0
    )

# Since the forecast is done on floor(len(test_cpu_series) / forecast_interval))
# times the test_cpu_series array needs to be also cut from len(forecast_array)
# to be able to calculate the rmse
test_cpu_series = test_cpu_series[:len(forecast_array)]

# Further on, we need to remove same amount of timesteps from start of the
# arrays what RNNs use as a input. This is done so that the rmse is
# calculated from the same exact part of the time series as is done with RNNs
forecast_array = forecast_array[config.get("history_size"):]
cpu_real = test_cpu_series[config.get("history_size"):]

# Clip out predictions to [0, 105] (max_value in training set for cpu was 104.4)
forecast_array = np.clip(forecast_array, 0, 105)

# Count rmse
rmse = mean_squared_error(cpu_real, forecast_array, squared=False)
rmse = round(rmse, 2)
print(f"RMSE in whole test set: {rmse}")


# Plot the forecast
if config.get("plot"):
    plt.plot(
        range(len(forecast_array)), forecast_array, alpha=0.4, label="forecast"
    )
    plt.plot(
        range(len(cpu_real)), cpu_real, alpha=0.4, label ="true CPU usage"
    )
    plt.legend(loc="upper left")
    plt.title((
        f"Arima{model.order} forecast for machine {config.get('machine_id')} "
        f"cpu_usage (RMSE = {rmse})"
    ))
    plt.show()