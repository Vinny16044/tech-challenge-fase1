"""
Módulo de monitoramento e logging dos experimentos da Fase 2.
Responsável: Paola
"""

import logging
import os
from datetime import datetime
from config import PHASE2_LOGS_DIR


def setup_logger(log_dir: str = PHASE2_LOGS_DIR):
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("experimento_%Y%m%d_%H%M%S.log")
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logging.info("Logger iniciado para experimentos da Fase 2.")

    return log_path


def log_experiment_result(model_name: str, params: dict, metrics: dict):
    logging.info("=" * 60)
    logging.info(f"Modelo: {model_name}")
    logging.info(f"Hiperparâmetros: {params}")
    logging.info(f"Accuracy: {metrics.get('accuracy')}")
    logging.info(f"Precision: {metrics.get('precision')}")
    logging.info(f"Recall: {metrics.get('recall')}")
    logging.info(f"F1-Score: {metrics.get('f1_score')}")
    logging.info(f"AUC-ROC: {metrics.get('auc_roc')}")