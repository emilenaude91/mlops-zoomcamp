import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_error
import numpy as np

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

# Set the tracking URI for the MLflow server
mlflow.set_tracking_uri("http://0.0.0.0:5001")
# Set the experiment name to use in MLflow
mlflow.set_experiment(EXPERIMENT_NAME)
# Enable MLflow autologging for scikit-learn
mlflow.sklearn.autolog()

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Start an MLflow run
    with mlflow.start_run():
        # Convert RF parameters to integers
        for param in RF_PARAMS:
            params[param] = int(params[param])

        # Train a RandomForestRegressor with the given parameters
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate the model on the validation and test sets
        val_rmse = np.sqrt(mean_squared_error(y_val, rf.predict(X_val)))  # Use np.sqrt to calculate RMSE
        mlflow.log_metric("val_rmse", val_rmse)  # Log validation RMSE
        test_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))  # Use np.sqrt to calculate RMSE
        mlflow.log_metric("test_rmse", test_rmse)  # Log test RMSE

        return mlflow.active_run().info.run_id  # Return the run ID of the current run

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):

    client = MlflowClient()

    # Retrieve the top_n model runs from the hyperparameter optimization experiment
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    
    best_run_id = None
    best_test_rmse = float('inf')
    
    for run in runs:
        run_id = train_and_log_model(data_path=data_path, params=run.data.params)
        
        # Get the test RMSE of the current run
        test_rmse = client.get_metric_history(run_id, "test_rmse")[0].value
        
        # Update the best run if the current run has a lower test RMSE
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_run_id = run_id

    # Register the best model with the lowest test RMSE
    if best_run_id is not None:
        model_uri = f"runs:/{best_run_id}/model"
        mlflow.register_model(model_uri=model_uri, name="random-forest-regressor")

if __name__ == '__main__':
    run_register_model()
