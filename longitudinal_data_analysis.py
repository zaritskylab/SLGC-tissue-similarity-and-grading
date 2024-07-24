"""
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from processing import process, aligned_representation
from correlation import correlation_analysis
from classification import classification_analysis

# Define current folder using this file
CWD = os.path.dirname(os.path.abspath(__file__))
# Define folder that contains the dhg dataset
DHG_PATH = os.path.join(CWD, "..", "data", "LONGITUDINAL")
# Define folder that contains raw data
DHG_RAW_DATA = os.path.join(DHG_PATH, "raw")
# Define folder to save aligned data
DHG_ALIGNED_DATA = os.path.join(DHG_PATH, "aligned")
# Define folder to save processed data
DHG_PROCESSED_DATA = os.path.join(DHG_PATH, "processed")
# Define file that contains dhg metadata
METADATA_PATH = os.path.join(DHG_PATH, "metadata.csv")
# Define path to save plots and results
FIGURES_PATH = os.path.join(CWD, "longitudinal")
# Define mass range start value
MZ_START = 50
# Define mass range end value
MZ_END = 1200
# Define mass resolution of the data
MASS_RESOLUTION = 0.025
# Define lock mass reference peak
LOCK_MASS_PEAK = 885.5498
# Define lock mass tol
LOCK_MASK_TOL = 0.3
# Define representative peaks
REPRESENTATIVE_PEAKS = [794.5, 834.5, 886.6]
# Define random seed
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':
  # Read metadata csv
  metadata_df = pd.read_csv(METADATA_PATH)
  """ Processing """
  # Loop over each unique msi imzML file
  for file_name in metadata_df.file_name.unique():
    # Define path to msi imzML file
    msi_path = os.path.join(DHG_RAW_DATA, f"{file_name}.imzML")
    # Define path to new msi imzML file after alignment
    output_path = os.path.join(DHG_ALIGNED_DATA, f"{file_name}.imzML")
    # Create output folder if doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Align MSI
    aligned_representation(
        msi_path, output_path, LOCK_MASS_PEAK, LOCK_MASK_TOL
    )
  
  # Loop over each ROI in data frame
  for index, roi in metadata_df.iterrows():
    # Define path to msi imzML file
    msi_path = os.path.join(DHG_ALIGNED_DATA, f"{roi.file_name}.imzML")
    # Define path to new msi imzML file after processing
    output_path = os.path.join(DHG_PROCESSED_DATA, f"{roi.sample_file_name}")
    # Create output folder if doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Process msi
    process(
        msi_path, output_path, roi.x_min, roi.x_max, roi.y_min, roi.y_max,
        MZ_START, MZ_END, MASS_RESOLUTION, REPRESENTATIVE_PEAKS
    )