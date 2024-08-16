import yaml
from os import path

def read_config(config_path):
    '''
    Returns all configuration options as a dictionary

        Parameters:
            config_path (str): Path to the config.yaml file

        Returns:
            config (dict): Dictionary containing all config information
    '''
    stream = open(config_path, 'r')
    config_dict = yaml.load(stream, Loader=yaml.SafeLoader)
    stream.close()
    return config_dict