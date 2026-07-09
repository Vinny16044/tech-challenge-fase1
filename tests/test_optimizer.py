import os
import sys

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from genetic.optimizer import train_and_evaluate_optimized_model


def test_train_and_evaluate_optimized_model():

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

    result = train_and_evaluate_optimized_model(
        model_name="Random Forest",
        params={
            "n_estimators": 10,
            "max_depth": 3
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    assert result is not None
    assert result["model_name"] == "Random Forest"
    assert "metrics" in result
    assert "recall" in result["metrics"]
    