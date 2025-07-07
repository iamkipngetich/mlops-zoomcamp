from prefect import flow, task
from duration_prediction import run  # Make sure this file is importable

@task
def run_training(year: int, month: int) -> str:
    run_id = run(year, month)
    return run_id

@flow(name="NYC Taxi Training Flow")
def training_flow(year: int, month: int):
    run_id = run_training(year, month)
    print(f"Completed MLflow run with ID: {run_id}")

if __name__ == "__main__":
    training_flow(year=2025, month=1)
