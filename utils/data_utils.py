import pandas as pd

def read_data(config, test_set=False):
    '''
    Returns dataset as pandas dataframe

        Parameters:
            config (dict): Configuration dictionary created with read_config()
            test_set (bool): Return test set? Defaults to returning training set

        Returns:
            df (DataFrame): Pandas DataFrame containing train / test set
    '''
    if not test_set:
        train_data_path = (
            f"{config.get('train_data_path')}/"
            f"{config.get('machine_id')}_train.pkl"
        )
        train_df = pd.read_pickle(train_data_path)
        return train_df
    else:
        test_data_path = (
            f"{config.get('test_data_path')}/"
            f"{config.get('machine_id')}_test.pkl"
        )
        test_df = pd.read_pickle(test_data_path)
        return test_df