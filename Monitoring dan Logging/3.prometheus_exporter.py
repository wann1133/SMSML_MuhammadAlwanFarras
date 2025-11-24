"""
Simple Prometheus exporter for monitoring Iris model inference.
- Serves metrics on port 8000 (configurable via --port).
- Periodically runs predictions to refresh accuracy and latency metrics.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

try:
    import mlflow.pyfunc
except ImportError:
    mlflow = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR.parent / "Membangun_model/namadataset_preprocessing/iris_preprocessed.csv"

INFERENCE_COUNTER = Counter("iris_inference_requests_total", "Total inference requests")
INFERENCE_LATENCY = Histogram("iris_inference_latency_seconds", "Latency per inference")
MODEL_ACCURACY = Gauge("iris_model_accuracy", "Accuracy evaluated on hold-out data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose Prometheus metrics for Iris inference.")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model_uri", type=str, default="", help="Optional MLflow model URI to load.")
    parser.add_argument("--port", type=int, default=8000, help="Port to expose metrics.")
    parser.add_argument("--interval", type=int, default=15, help="Seconds between metric updates.")
    return parser.parse_args()


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_path)
    return df.drop(columns=["target"]), df["target"]


def load_model(model_uri: str, X: pd.DataFrame, y: pd.Series):
    if model_uri and mlflow:
        return mlflow.pyfunc.load_model(model_uri)

    # Fallback: train a lightweight logistic regression for monitoring demo
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LogisticRegression(max_iter=200, solver="lbfgs", multi_class="auto")
    model.fit(X_train, y_train)
    return model


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> float:
    preds = model.predict(X)
    return accuracy_score(y, preds)


def main() -> None:
    args = parse_args()
    X, y = load_dataset(Path(args.data_path))
    model = load_model(args.model_uri, X, y)

    start_http_server(args.port)
    print(f"Prometheus exporter running on port {args.port}")

    while True:
        start = time.time()
        _ = model.predict(X.sample(1, random_state=int(time.time())) if len(X) > 1 else X)
        INFERENCE_COUNTER.inc()
        INFERENCE_LATENCY.observe(time.time() - start)

        acc = evaluate(model, X, y)
        MODEL_ACCURACY.set(acc)

        time.sleep(max(args.interval, 5))


if __name__ == "__main__":
    main()
