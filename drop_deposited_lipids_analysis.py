"""Chip types data analysis
The script should be ran and will read imzML files, process data and create
relevant plot such as feature counts per Chip types.

"""
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from processing import common_representation, aligned_representation


def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define folder that contains the revision chip type dataset
  CHIP_TYPES_PATH = Path(
      os.path.join(CWD, "..", "data", "DROP_DEPOSITED_LIPIDS")
  )
  # Define folder that contains raw data
  RAW_DATA = CHIP_TYPES_PATH.joinpath("raw")
  # Define folder to save aligned data
  ALIGNED_DATA = CHIP_TYPES_PATH.joinpath("aligned")
  # Define folder to save processed data
  PROCESSED_DATA = CHIP_TYPES_PATH.joinpath("processed")
  # Define file that contains dhg metadata
  METADATA_PATH = CHIP_TYPES_PATH.joinpath("metadata.csv")
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

  # Define random seed
  SEED = 42
  random.seed(SEED)
  np.random.seed(SEED)
  # Read metadata csv
  metadata_df = pd.read_csv(METADATA_PATH)
  metadata_df["sample_number"] = metadata_df.sample_file_name.apply(
      lambda s: s.split("_")[0]
  )

  # Loop over each unique msi imzML file
  for file_name in metadata_df.file_name.unique():
    # Define path to msi imzML file
    msi_path = os.path.join(RAW_DATA, f"{file_name}.imzML")
    # Define path to new msi imzML file after alignment
    output_path = os.path.join(ALIGNED_DATA, f"{file_name}.imzML")
    # Create output folder if doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # Align MSI
    aligned_representation(msi_path, output_path, LOCK_MASS_PEAK, LOCK_MASK_TOL)

  # Loop over each ROI in data frame
  for _, roi in metadata_df.iterrows():
    # Define path to msi imzML file
    msi_path = os.path.join(RAW_DATA, f"{roi.file_name}.imzML")
    # Define path to new msi imzML file after processing
    output_path = os.path.join(PROCESSED_DATA, f"{roi.sample_file_name}")
    # Create output folder if doesn't exist
    Path(output_path).mkdir(parents=True, exist_ok=True)
    # Process msi
    common_representation(
        msi_path, output_path, roi.x_min, roi.x_max, roi.y_min, roi.y_max,
        MZ_START, MZ_END, MASS_RESOLUTION / 2
    )


if __name__ == '__main__':
  main()
