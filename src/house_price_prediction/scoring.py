import argparse
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from logger_functions import configure_logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train import split_DV_IV

HOUSING_MODEL_FOLDER = "artifacts"
HOUSING_DATA_FOLDER = "data/processed"
HOUSING_OUTPUT_FOLDER = "notebooks/results"

logger = logging.getLogger(__name__)
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000/"


def initialize_parser():
    """
    Initialize the argument parser.

    Returns:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-data-folder",
        help="Specify input data folder",
        default=HOUSING_DATA_FOLDER,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--input-model-folder",
        help="Specify input model folder",
        default=HOUSING_MODEL_FOLDER,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--output-folder",
        help="Specify output folder",
        default=HOUSING_OUTPUT_FOLDER,
        type=str,
        required=False,
    )

    parser.add_argument(
        "--log-level",
        help="Logger level default: %(default)s",
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
        help="Print to console default: %(default)s",
        default=True,
        action="store_false",
    )

    args = parser.parse_args()

    if args.output_folder is not None:
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)

    return args


def get_score():
    """
    Calculate the scores of a trained regression model on the test dataset.
    """
    global logger
    experiment_id = mlflow.create_experiment("Scoring of trained model")

    with mlflow.start_run(
        run_name="Parent_run",
        experiment_id=experiment_id,
        description="Scoring of trained model",
    ):

    args = initialize_parser()
    logger = configure_logger(
        logger=logger,
        log_file=args.log_path,
        console=args.no_console_log,
        log_level=args.log_level,
    )
    logger.info("Started Scoring")

    logger.info("Loaded train and test datasets")
    train_df = pd.read_csv(args.input_data_folder + "/train.csv")
    test_df = pd.read_csv(args.input_data_folder + "/test.csv")

    X_train, y_train = split_DV_IV(train_df, "median_house_value")
    X_test, y_test = split_DV_IV(test_df, "median_house_value")

    logger.info("Loading trained feature transformer")
    feature_transformer = joblib.load(
        args.input_model_folder + "/feature_transformer.joblib"
    )

    X_train_trans = feature_transformer.transform(X_train)
    X_test_trans = feature_transformer.transform(X_test)

    metrics_df = pd.DataFrame(columns=["mse", "rmse", "mae"])

    logger.info("Loading trained model")
    final_model = joblib.load(args.input_model_folder + "/final_model.joblib")
    mlflow.log_artifact(args.input_model_folder + "/final_model.joblib")

    y_train_pred = final_model.predict(X_train_trans)
    y_test_pred = final_model.predict(X_test_trans)

    metrics_df.loc["train", "mse"] = mean_squared_error(y_train, y_train_pred)
    metrics_df.loc["train", "rmse"] = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )
    metrics_df.loc["train", "mae"] = mean_absolute_error(y_train, y_train_pred)
    metrics_df.loc["test", "mse"] = mean_squared_error(y_test, y_test_pred)
    metrics_df.loc["test", "rmse"] = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )
    metrics_df.loc["test", "mae"] = mean_absolute_error(y_test, y_test_pred)
    logger.info(f"Metrics from the trained model: \n{metrics_df}")

    metrics_df.to_csv(HOUSING_OUTPUT_FOLDER + "/metrics.csv")
    logger.info("Completed saving the metrics to the file")


if __name__ == "__main__":
    get_score()
