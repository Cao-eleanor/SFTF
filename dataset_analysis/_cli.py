"""Shared command-line argument parsing for the dataset-analysis scripts.

Every script in this folder produces one table or figure from Chapter 3
(Section III) of the paper. This module centralizes the common flags so the
scripts share one consistent CLI instead of re-declaring the same options.
"""

import argparse
from pathlib import Path

# Resolve paths relative to this module's location so the code runs from any
# working directory and on any machine (portable for redistribution).
DATASET_ANALYSIS_DIR = Path(__file__).resolve().parent        # SFTF/dataset_analysis/
DATA_DIR = DATASET_ANALYSIS_DIR.parent / 'model' / 'data'     # SFTF/model/data/
RAW_DATA_DIR = DATA_DIR / 'raw_data'
OUTPUT_DIR = DATASET_ANALYSIS_DIR / 'outputs'


def arguments(description, sentiment=False):
    """Build and parse the shared command-line arguments.

    Args:
        description: one-line summary shown in --help.
        sentiment: add the --sentiment scoring-method flag.

    Returns:
        argparse.Namespace exposing data_dir, output_dir, dpi, show, and
        (when sentiment=True) sentiment.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--data-dir', type=Path, default=RAW_DATA_DIR,
                        help='Directory holding the input .xlsx tables.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help='Directory receiving output files.')
    parser.add_argument('--dpi', type=int, default=600,
                        help='Output figure resolution (default: 600).')
    parser.add_argument('--show', action='store_true',
                        help='Display figures instead of closing them.')
    if sentiment:
        parser.add_argument('--sentiment', choices=('keyword', 'snownlp'),
                            default='keyword',
                            help='Sentiment scoring method (default: keyword).')
    return parser.parse_args()
