from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, train_test_split


DATA_PATH = Path(__file__).resolve().parent / "namadataset_preprocessing/iris_preprocessed.csv"
TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual MLflow logging with hyperparameter tuning for the Iris dataset."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DATA_PATH,
        help="Path to the preprocessed dataset CSV.",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Hold-out ratio for evaluation.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    return df.drop(columns=["target"]), df["target"]


def evaluate_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro"),
        "recall_macro": recall_score(y_test, preds, average="macro"),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }


def main() -> None:
    args = parse_args()
    X, y = load_data(Path(args.data_path))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    param_grid = ParameterGrid(
        {
            "n_estimators": [80, 120, 180],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 4],
            "max_features": ["sqrt", None],
        }
    )

    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("iris-tuning-manual")

    best_model = None
    best_params = None
    best_metrics = None

    with mlflow.start_run(run_name="rf-grid-search") as parent_run:
        mlflow.log_param("data_path", str(Path(args.data_path).resolve()))
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        for idx, params in enumerate(param_grid):
            with mlflow.start_run(run_name=f"trial-{idx}", nested=True):
                candidate = RandomForestClassifier(random_state=args.random_state, **params)
                metrics = evaluate_model(candidate, X_train, X_test, y_train, y_test)
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                if best_metrics is None or metrics["f1_macro"] > best_metrics["f1_macro"]:
                    best_model = candidate
                    best_params = params
                    best_metrics = metrics

        if best_model is not None:
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
            mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            print("Best params:", best_params)
            print("Best metrics:", best_metrics)
        else:
            print("No model was trained; check dataset or parameter grid.")


if __name__ == "__main__":
    main()
