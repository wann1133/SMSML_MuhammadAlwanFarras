"""
Automated preprocessing pipeline converted from the experiment notebook.
Steps:
- drop duplicates
- impute missing numeric values with the median
- scale numeric features
- encode target labels
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


RAW_DEFAULT = Path(__file__).resolve().parent.parent / "namadataset_raw/iris_raw.csv"
OUTPUT_DEFAULT = Path(__file__).resolve().parent / "namadataset_preprocessing/iris_preprocessed.csv"


def load_raw_dataset(raw_path: Path = RAW_DEFAULT) -> pd.DataFrame:
    """Load the raw CSV into a DataFrame."""
    df = pd.read_csv(raw_path)
    return df


def build_preprocess_pipeline(feature_columns: Iterable[str]) -> ColumnTransformer:
    """Create the preprocessing pipeline for numeric features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_transformer, list(feature_columns))],
        remainder="drop",
    )


def encode_target(target: pd.Series) -> Tuple[pd.Series, LabelEncoder]:
    """Encode string labels into integers."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(target)
    return encoded, encoder


def run_preprocessing(
    raw_path: Path = RAW_DEFAULT,
    output_path: Path = OUTPUT_DEFAULT,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, ColumnTransformer, LabelEncoder]:
    """Execute the preprocessing steps and persist the processed dataset."""
    raw_path = Path(raw_path)
    output_path = Path(output_path)

    df = load_raw_dataset(raw_path)
    df = df.drop_duplicates()

    feature_cols = [col for col in df.columns if col != "target"]
    target = df["target"]

    preprocess = build_preprocess_pipeline(feature_cols)
    X_processed = preprocess.fit_transform(df[feature_cols])
    processed_df = pd.DataFrame(
        X_processed, columns=[f"feature_{i+1}" for i in range(X_processed.shape[1])]
    )

    y_encoded, encoder = encode_target(target)
    processed_df["target"] = y_encoded

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(output_path, index=False)

    if verbose:
        print(f"Saved preprocessed dataset to {output_path} with shape {processed_df.shape}")

    return processed_df, preprocess, encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate preprocessing for the Iris dataset.")
    parser.add_argument(
        "--raw_path",
        type=Path,
        default=RAW_DEFAULT,
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=OUTPUT_DEFAULT,
        help="Path to save the preprocessed CSV dataset.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress logging output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocessing(
        raw_path=args.raw_path,
        output_path=args.output_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
