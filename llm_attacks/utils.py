import os
import sys
import datetime
import logging


def setup_logging(log_root, log_name_prefix):
    current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_root, f'{log_name_prefix}_{current_date}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger, file_handler


def setup_paths():
    workspace_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    sys.path.insert(0, workspace_root)
    data_root = os.path.join(workspace_root, "data")
    log_root = os.path.join(workspace_root, "./")
    os.makedirs(log_root, exist_ok=True)
    return workspace_root, data_root, log_root


def get_model_path():
    model_path = os.environ.get("HF_MODEL_PATH")
    return "" if model_path is None else model_path