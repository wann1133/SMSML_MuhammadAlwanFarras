from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    import mlflow.pyfunc
except ImportError:
    mlflow = None  # type: ignore


DEFAULT_DATA = Path(__file__).resolve().parent.parent / "Membangun_model/namadataset_preprocessing/breast_cancer_preprocessed.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple inference helper for the Breast Cancer model.")
    parser.add_argument("--model_uri", type=str, default="", help="Optional MLflow model URI or local path.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=DEFAULT_DATA,
        help="Fallback dataset used for training/picking default features.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=float,
        metavar=("f1", "f2", "..."),
        help="Optional feature values; provide 30 numbers if supplied.",
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    return df.drop(columns=["target"]), df["target"]


def load_model(model_uri: str, X: pd.DataFrame, y: pd.Series):
    if model_uri and mlflow:
        return mlflow.pyfunc.load_model(model_uri)
    model = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto")
    model.fit(X, y)
    return model


def build_feature_frame(feature_values: Sequence[float], template: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([feature_values], columns=template.columns)


def main() -> None:
    args = parse_args()
    X, y = load_dataset(Path(args.data_path))

    model = load_model(args.model_uri, X, y)

    if args.features:
        features = build_feature_frame(args.features, X)
    else:
        features = X.sample(1, random_state=42)
        print("No features provided; using a sample row from the dataset.")

    preds = model.predict(features)
    pred_class = int(preds[0])
    label = "benign" if pred_class == 1 else "malignant"
    print(f"Predicted class: {pred_class} ({label})")


if __name__ == "__main__":
    main()
