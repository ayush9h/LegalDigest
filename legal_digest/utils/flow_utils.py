import mlflow
import mlflow.pytorch

def init_flow():

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("legal-digest-t5-small")
