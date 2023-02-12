"""_summary_
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from processing import process
from correlation import correlation_analysis

# Define figure defaults
sns.set_style("white")
sns.set_context("paper")

# Define folder that contains the dhg dataset
DHG_PATH = "../DHG/"
# Define folder that contains raw data
DHG_RAW_DATA = f"{DHG_PATH}/raw"
# Define folder to save processed data
DHG_PROCESSED_DATA = f"{DHG_PATH}/processed"
# Define file that contains dhg metadata
METADATA_PATH = f"{DHG_PATH}/metadata.csv"
# Define mass range start value
MZ_START = 50
# Define mass range end value
MZ_END = 1200
# Define mass resolution of the data
MASS_RESOLUTION = 0.025
# Define representative peaks
REPRESENTATIVE_PEAKS = [794.5, 834.5, 886.6]

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
    """
    process(
        msi_path, output_path, roi.x_min, roi.x_max, roi.y_min, roi.y_max,
        MZ_START, MZ_END, MASS_RESOLUTION, REPRESENTATIVE_PEAKS
    )
    """
  correlation_analysis(DHG_PROCESSED_DATA)
  """ Correlation analysis"""
  """ Classification analysis"""