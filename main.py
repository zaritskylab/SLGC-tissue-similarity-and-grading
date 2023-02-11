"""_summary_
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from processing.process import process_msi

# Define figure defaults
sns.set_style("white")
sns.set_context("paper")

# Define folder that contains the dhg dataset
DHG_PATH = "./DHG/"
# Define folder that contains level 0 data
LEVEL_0_PATH = f"{DHG_PATH}/level_0"
# Define folder to save level 1 data
LEVEL_1_PATH = f"{DHG_PATH}/level_1"
# Define file that contains dhg metadata
METADATA_PATH = f"{DHG_PATH}/metadata.csv"
# Define path to save figures
FIGURES_PATH = "./figures/"
# Define mass range start value
MZ_START = 50
# Define mass range end value
MZ_END = 1200
# Define mass resolution of the data
MASS_RESOLUTION = 0.025

if __name__ == '__main__':
  # Read metadata csv
  metadata_df = pd.read_csv(METADATA_PATH)
  """ Processing """
  # Loop over each ROI in data frame
  for index, roi in metadata_df.iterrows():
    # Define path to msi imzML file
    msi_path = os.path.join(LEVEL_0_PATH, f"{roi.file_name}.imzML")
    # Define path to new msi imzML file after processing
    roi_path = os.path.join(LEVEL_1_PATH, f"{roi.sample_file_name}.imzML")
    # Process msi
    process_msi(
        msi_path, roi_path, roi.x_min, roi.x_max, roi.y_min, roi.y_max,
        MZ_START, MZ_END, MASS_RESOLUTION / 2
    )
  """ Correlation analysis"""
  """ Classification analysis"""