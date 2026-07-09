import sys
import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from genetic.experiments import run_phase2_experiments


def test_run_phase2_experiments(tmp_path):
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    baseline_results = pd.DataFrame({
        "accuracy": [0.80],
        "recall": [0.80],
        "f1_score": [0.80]
    }, index=["Random Forest"])

    df_result = run_phase2_experiments(
        dataset_name="Teste",
        baseline_results=baseline_results,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=str(tmp_path)
    )

    assert df_result is not None
    assert not df_result.empty
    assert "dataset" in df_result.columns
    assert "experiment" in df_result.columns
    assert "model" in df_result.columns
    assert "optimized_recall" in df_result.columns