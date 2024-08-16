import pandas as pd
import random
from statistics import mean
from math import floor
from os import path
from tqdm import tqdm
from utils import yaml_utils

# Load in configs from config.yaml which needs to be in same directory as this
# python file
config = yaml_utils.read_config(path.dirname(path.abspath(__file__)) + \
    "/config.yaml")

# Lets first create empty dataframe with columns, later we append to this while
# looping over csv files
df = pd.DataFrame(columns = [
    "machine_id", "timestamp", "cpu_usage", "memory_usage",
    "disk_read", "disk_write", "net_receive", "net_transmit"
    ])

# Loop through csv files and append data to df
print("Generating pandas dataframe from CSV data...")
for i in tqdm(range(1,1251)):
    # Read one machine data
    temp_df = pd.read_csv(
        f"{config.get('bitbrains_data_path')}/{i}.csv",
        sep=";\t", engine="python"
    )

    # Drop unnecessary columns
    temp_df.drop(
        columns=
            ['CPU cores', 'CPU capacity provisioned [MHZ]',
            'CPU usage [MHZ]', 'Memory capacity provisioned [KB]'],
        inplace=True
    )

    # Rename columns
    temp_df.columns = [
        "timestamp", "cpu_usage", "memory_usage", "disk_read",
        "disk_write", "net_receive", "net_transmit"
    ]

    # Add machine_id column
    machine_id_col = [i] * len(temp_df.index)
    temp_df.insert(0, "machine_id", machine_id_col)

    # Append one machine data to df holding all data
    df = pd.concat([df, temp_df], ignore_index=True)


# Change datatypes
df["machine_id"] = pd.to_numeric(df["machine_id"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')

# Set timestamp as index
df = df.set_index("timestamp")

# Group by machine_id
grouped_df = df.groupby("machine_id")

# Loop over grouped df to find machines with mean cpu_usage > 30
print("Finding wanted machines...")
wanted_machines = []
for machine, machine_df in grouped_df:
    if mean(machine_df["cpu_usage"].values) > 30:
        wanted_machines.append(machine)

# Choose 5 random machines from wanted machines
random.seed(2207)
selected_machines = random.sample(wanted_machines, 5)
print(f"Selected machines: {selected_machines}")

# Now take these machines and create [train, test] -sets with [75, 25]%
split = 0.75
for machine in selected_machines:
    machine_df = grouped_df.get_group(machine)
    n = len(machine_df.index)
    train_n = floor(split * n)
    test_n = n - train_n
    train_set = machine_df.head(train_n)
    test_set = machine_df.tail(test_n)

    # Save train, test sets to disk as separate files
    train_set.to_pickle(
        f"{config.get('train_data_path')}/{machine}_train.pkl", protocol=3
    )
    test_set.to_pickle(
        f"{config.get('test_data_path')}/{machine}_test.pkl", protocol=3
    )
print("Train and test sets has been created")