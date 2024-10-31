"""Drop deposited lipids analysis.

This script contains the code to analyze the drop deposited lipids dataset and 
create relevant plot such as best fit plot.

"""
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from pyimzml.ImzMLParser import ImzMLParser
from utils import read_msi
from processing import common_representation, aligned_representation
from analysis.esi_data_analysis import plot_spectras_best_fit_hex


def get_spectras(
    imgs: Dict[str, ImzMLParser],
    mzs: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
  """Get the spectras for each image.

  Args:
    parsers (Dict[str, ImzMLParser]): A dictionary containing parsers.
    mzs (np.ndarray): The m/z values.

  Returns:
    Dict[str, Tuple[np.ndarray, np.ndarray]]: A dict containing the 
        spectras for each image.
  
  """
  # Define m/z filter
  mzs_filter = (mzs > 600) & (mzs < 900)

  # Dictionary to store the spectras
  spectras = {}
  # Loop through each image
  for name, img in imgs.items():
    # Capitalize the name if it contains "flat"
    if "flat" in name:
      name = name.capitalize()

    # Get mean intensity for each m/z value in the filter
    spectras[name] = (mzs[mzs_filter], img[:, :, mzs_filter].mean(axis=(0, 1)))
  return spectras


def main():
  """Function containing main code"""
  # Define current folder using this file
  CWD = Path(os.path.dirname(os.path.abspath(__file__)))
  # Define folder that contains the drop deposited lipids dataset
  DROP_DEPOSITED_LIPIDS_PATH = Path(
      os.path.join(CWD, "..", "..", "data", "DROP_DEPOSITED_LIPIDS")
  )
  # Define folder that contains raw data
  RAW_DATA = DROP_DEPOSITED_LIPIDS_PATH.joinpath("raw")
  # Define folder to save aligned data
  ALIGNED_DATA = DROP_DEPOSITED_LIPIDS_PATH.joinpath("aligned")
  # Define folder to save processed data
  PROCESSED_DATA = DROP_DEPOSITED_LIPIDS_PATH.joinpath("processed")
  # Define file that contains metadata
  METADATA_PATH = DROP_DEPOSITED_LIPIDS_PATH.joinpath("metadata.csv")
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

  # Define path to save figures
  PLOT_PATH = CWD / "drop_deposited_lipids"
  # Create dirs
  PLOT_PATH.mkdir(parents=True, exist_ok=True)

  # Define dictionary to store images
  imgs = {}

  # Loop over each processed file
  for file in (DROP_DEPOSITED_LIPIDS_PATH / "processed").iterdir():
    # Read msi
    with ImzMLParser(file / "common_representation.imzML") as p:
      # Get mzs and img
      mzs, img = read_msi(p)
      # Store img in dictionary
      imgs[file.stem] = img

  # Convert to spectras dictionary for best fit hex plot
  spectras = get_spectras(imgs, mzs)

  # Plot best fit hex
  plot_spectras_best_fit_hex(spectras, PLOT_PATH, lambda x: x.replace("_", " "))


if __name__ == '__main__':
  main()
