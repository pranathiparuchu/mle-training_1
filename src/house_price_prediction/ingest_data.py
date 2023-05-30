import argparse
import logging
import os
import os.path as path
import tarfile
import urllib.request
from urllib.error import URLError

import numpy as np
import pandas as pd
from logger_functions import configure_logger
from sklearn.model_selection import StratifiedShuffleSplit

HOUSING_ROOT_PATH = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"
HOUSING_OUTPUT_FOLDER = "data/processed"
HOUSING_DATA_FOLDER = "data"

# Configure logging
configure_logger()
logger = logging.getLogger(__name__)


def fetch_housing_data(
    housing_url=HOUSING_ROOT_PATH, housing_path=HOUSING_DATA_FOLDER
):
    """
    Fetches the housing data from the given URL and saves it to the specified path.

    Args:
        housing_url (str): The URL of the housing data.
        housing_path (str): The path where the data will be saved.
    """
    try:
        logger.info("Fetching housing data...")
        os.makedirs(housing_path + "/raw", exist_ok=True)
        tgz_path = os.path.join(housing_path + "/raw", "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        logger.info("Data fetched sucessfully")
    except URLError:
        logger.exception(
            f"Unable to fetch the dataset from the provided url {DEFAULT_ROOT_PATH}"
        )
        exit()

    try:
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path + "/raw")
        housing_tgz.close()
        logger.info("Housing data extracted and saved.")

    except FileNotFoundError:
        logger.exception("Dataset is not found")


def load_housing_data(housing_path=HOUSING_DATA_FOLDER):
    """
    Loads the housing data from the specified path and returns it as a pandas DataFrame.

    Args:
        housing_path (str): The path where the data is located.

    Returns:
        pandas.DataFrame: The housing data.
    """
    logger.info("Loading housing data...")
    csv_path = os.path.join(housing_path, "raw", "housing.csv")
    data = pd.read_csv(csv_path)
    logger.info("Housing data loaded.")
    return data


def add_new_columns(df):
    """
    Adds new columns to the housing DataFrame.

    Args:
        df (pandas.DataFrame): The housing data.

    Returns:
        pandas.DataFrame: The housing data with new columns added.
    """
    logger.info("Adding new columns to the housing data...")
    housing = df.copy()
    housing["rooms_per_household"] = (
        housing["total_rooms"] / housing["households"]
    )
    housing["bedrooms_per_room"] = (
        housing["total_bedrooms"] / housing["total_rooms"]
    )
    housing["population_per_household"] = (
        housing["population"] / housing["households"]
    )
    logger.info("New columns added.")
    return housing


def split_data(df):
    """
    Splits the housing data into training and testing sets.

    Args:
        df (pandas.DataFrame): The housing data.

    Returns:
        tuple: A tuple containing the stratified training set and the stratified testing set.
    """
    logger.info("Splitting housing data into training and testing sets...")
    housing = df.copy()

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    logger.info("Housing data split.")
    return strat_train_set, strat_test_set


def initialize_parser():
    """
    Parses command-line arguments so they can be used in the code.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-folder",
        help="Specify output folder",
        default=HOUSING_OUTPUT_FOLDER,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--log-level",
        help="Logger level HOUSE: %(HOUSE)s",
        default="DEBUG",
        choices=["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"],
        required=False,
    )

    parser.add_argument(
        "--log-path",
        help="Path of the logger file",
        default=None,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--no-console-log",
        help="Print to console HOUSE: %(HOUSE)s",
        default=True,
        action="store_false",
    )

    args = parser.parse_args()

    return args


def get_data():
    """
    Fetches and processes the housing data.

    """
    global logger
    args = initialize_parser()
    logger = configure_logger(
        logger=logger,
        log_file=args.log_path,
        console=args.no_console_log,
        log_level=args.log_level,
    )
    if not os.path.exists(HOUSING_DATA_FOLDER + "/raw"):
        logger.info(
            f'Directory "{HOUSING_DATA_FOLDER + "/raw"}" not found so creating the same'
        )
        os.makedirs(HOUSING_DATA_FOLDER + "/raw")

    if len(os.listdir(HOUSING_DATA_FOLDER + "/raw")) == 0:
        fetch_housing_data()
        housing = load_housing_data(housing_path=HOUSING_DATA_FOLDER)
        housing = add_new_columns(housing)
    else:
        housing = load_housing_data()
        housing = add_new_columns(housing)

    train_set, test_set = split_data(housing)

    if not os.path.exists(args.output_folder):
        logger.info(
            f"Directory '{args.output_folder}' not found so creating the same"
        )
        os.makedirs(args.output_folder)

    train_set.to_csv(args.output_folder + "/train.csv", index=False)
    test_set.to_csv(args.output_folder + "/test.csv", index=False)


if __name__ == "__main__":
    get_data()
