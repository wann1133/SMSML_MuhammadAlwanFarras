from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).resolve().parent / "namadataset_preprocessing/iris_preprocessed.csv"
TRACKING_DIR = Path(__file__).resolve().parent / "mlruns"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a basic Iris classifier with MLflow autologging.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DATA_PATH,
        help="Path to the preprocessed dataset CSV.",
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Hold-out ratio for evaluation.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--max_iter", type=int, default=200, help="Max iterations for Logistic Regression.")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization strength for Logistic Regression.")
    return parser.parse_args()


def load_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def train_model(args: argparse.Namespace) -> str:
    X, y = load_data(Path(args.data_path))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    mlflow.set_experiment("iris-basic-autolog")
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run(run_name="logreg-basic") as run:
        model = LogisticRegression(
            max_iter=args.max_iter,
            C=args.C,
            penalty="l2",
            solver="lbfgs",
            multi_class="auto",
            random_state=args.random_state,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision_macro": precision_score(y_test, preds, average="macro"),
            "recall_macro": recall_score(y_test, preds, average="macro"),
            "f1_macro": f1_score(y_test, preds, average="macro"),
        }

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print("\nClassification report:\n", classification_report(y_test, preds))
        return run.info.run_id


def main() -> None:
    args = parse_args()
    run_id = train_model(args)
    print(f"Run completed. MLflow run id: {run_id}")


if __name__ == "__main__":
    main()
