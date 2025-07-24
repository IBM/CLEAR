import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from clear_eval.pipeline.constants import SCORE_COL
logger = logging.getLogger(__name__)



def resolve_data_path(input_path: str) -> Path:
    input_path = Path(input_path).expanduser()

    if input_path.exists():
        return str(input_path.resolve())

    project_root = Path(__file__).resolve().parent.parent
    fallback_path = project_root / "sample_data" / input_path
    if fallback_path.exists():
        return str(fallback_path)
    else:
        raise FileNotFoundError(f"Could not find data file: {input_path}")



def ensure_dir(directory_path):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
        except OSError as e:
            logger.error(f"Error creating directory {directory_path}: {e}")
            sys.exit(1)  # Exit if we can't create essential directory


def load_dataframe_from_cache(path, expected_rows=None):
    """Loads a DataFrame from cache if RESUME_ENABLED and file exists and is valid."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if expected_rows is not None and len(df) != expected_rows:
                logger.info(f"Cache miss for {path}: Expected {expected_rows} rows, got {len(df)}. Recomputing.")
                return None
            logger.info(f"Loaded DataFrame from cache: {path}")
            # Convert potential string "NA" back to actual pd.NA for score columns
            if SCORE_COL in df.columns:
                df[SCORE_COL] = df[SCORE_COL].astype('Float64')
            return df
        except Exception as e:
            logger.error(f"Error loading DataFrame from cache {path}: {e}. Recomputing.")
            return None
    logger.info(f"Cache miss for {path}. Recomputing.")
    return None


def save_dataframe_to_cache(df, path):
    """Saves a DataFrame to cache."""
    try:
        df.to_csv(path, index=False)
        logger.info(f"Saved DataFrame to cache: {path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to cache {path}: {e}")


def load_json_from_cache(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded JSON from cache: {path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON from cache {path}: {e}. Recomputing.")
            return None
    return None


def save_json_to_cache(data, path):
    """Saves data to JSON cache."""
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON to cache: {path}")
    except Exception as e:
        logger.error(f"Error saving JSON to cache {path}: {e}")


