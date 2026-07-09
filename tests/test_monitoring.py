import sys
import os
import logging

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

from monitoring import setup_logger, log_experiment_result


def test_setup_logger_creates_log_file(tmp_path):
    log_dir = tmp_path / "logs"

    log_path = setup_logger(log_dir=str(log_dir))

    assert os.path.exists(log_path)
    assert log_path.endswith(".log")


def test_log_experiment_result(tmp_path):
    log_dir = tmp_path / "logs"

    log_path = setup_logger(log_dir=str(log_dir))

    metrics = {
        "accuracy": 0.90,
        "precision": 0.88,
        "recall": 0.92,
        "f1_score": 0.89,
        "auc_roc": 0.94
    }

    log_experiment_result(
        model_name="Random Forest",
        params={"n_estimators": 100},
        metrics=metrics
    )

    logging.shutdown()

    assert os.path.exists(log_path)

    with open(log_path, "r", encoding="utf-8") as file:
        content = file.read()

    assert "Random Forest" in content
    assert "n_estimators" in content
    assert "Recall" in content