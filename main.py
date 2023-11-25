""" Main script
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from processing import process
from correlation import correlation_analysis
from classification import classification_analysis

# Define current folder using this file
CWD = os.path.dirname(os.path.abspath(__file__))
# Define folder that contains the dhg dataset
DHG_PATH = os.path.join(CWD, "..", "data", "DHG")
# Define folder that contains raw data
DHG_RAW_DATA = os.path.join(DHG_PATH, "raw")
# Define folder to save processed data
DHG_PROCESSED_DATA = os.path.join(DHG_PATH, "processed")
# Define file that contains dhg metadata
METADATA_PATH = os.path.join(DHG_PATH, "metadata.csv")
# Define path to save plots and results
FIGURES_PATH = os.path.join(CWD, "figures")
# Define mass range start value
MZ_START = 50
# Define mass range end value
MZ_END = 1200
# Define mass resolution of the data
MASS_RESOLUTION = 0.025
# Define representative peaks
REPRESENTATIVE_PEAKS = [794.5, 834.5, 886.6]
# Define random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

if __name__ == '__main__':
  # Read metadata csv
  metadata_df = pd.read_csv(METADATA_PATH)
  """ Processing """
  # Loop over each ROI in data frame
  for index, roi in metadata_df.iterrows():
    # Define path to msi imzML file
    msi_path = os.path.join(DHG_RAW_DATA, f"{roi.file_name}.imzML")
    # Define path to new msi imzML file after processing
    output_path = os.path.join(DHG_PROCESSED_DATA, f"{roi.sample_file_name}")
    # Create output folder if doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Process msi
    process(
        msi_path, output_path, roi.x_min, roi.x_max, roi.y_min, roi.y_max,
        MZ_START, MZ_END, MASS_RESOLUTION, REPRESENTATIVE_PEAKS
    )
  """ Correlation analysis"""
  # Define path to save correlations
  output_path = os.path.join(FIGURES_PATH, "correlations")
  # Create output folder if doesn't exist
  Path(output_path).mkdir(parents=True, exist_ok=True)
  # Correlation analysis
  correlation_analysis(DHG_PROCESSED_DATA, output_path)
  """ Classification analysis"""
  # Define binary classification label
  metadata_df["label"] = (metadata_df["who_grade"] > 2).astype(int)
  # Define path to save correlations
  output_path = os.path.join(FIGURES_PATH, "classification")
  # Create output folder if doesn't exist
  Path(output_path).mkdir(parents=True, exist_ok=True)
  classification_analysis(DHG_PROCESSED_DATA, output_path, metadata_df)
