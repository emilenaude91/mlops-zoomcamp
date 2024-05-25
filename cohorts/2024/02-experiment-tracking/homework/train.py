import os
import pickle
import click
import mlflow  # Added import for mlflow
import mlflow.sklearn  # Added import for mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set tracking URI

mlflow.set_tracking_uri("http://13.53.123.37:5000")

# Set experiment 

mlflow.set_experiment("nyc_taxi_experiment")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)

def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run():  # Added MLflow start run context manager

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        
        # Log model and parameters
        mlflow.log_param("max_depth", 10)  # Log the max_depth parameter
        mlflow.log_param("random_state", 0)  # Log the random_state parameter
        mlflow.log_metric("rmse", rmse)  # Log the RMSE metric
        mlflow.sklearn.log_model(rf, "model")  # Log the trained model
        mlflow.log_param("min_samples_split", rf.min_samples_split)  # Log the min_samples_split parameter


if __name__ == '__main__':
    run_train()
