import os
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy import stats
from pyimzml.ImzMLParser import ImzMLParser
from utils import read_msi


def correlation(
    processed_path: str, before_correction: bool, sample_type_a: str,
    sample_type_b: str, spectra_type_a: str, spectra_type_b: str
):
  #
  representatives = {}
  #
  name = (
      "common_representation.imzML"
      if before_correction else "meaningful_signal.imzML"
  )
  #
  for folder in os.listdir(processed_path):
    #
    if ~os.path.isdir(os.path.join(processed_path, folder)):
      continue
    #
    with ImzMLParser(os.path.join(processed_path, folder, name)) as reader:
      #
      segment_image = np.load(
          os.path.join(processed_path, folder, name, "segmentation.npy")
      )
      #
      mzs, data = read_msi(reader)
      #
      if sample_type_a == "tissue":
        representatives


def get_representative_spectras(processed_path):
  # Define dict to store all representative spectras
  representatives = {
      "common_representation": {
          "tissue": {},
          "background": {}
      },
      "meaningful_signal": {
          "tissue": {},
          "background": {}
      }
  }

  # Loop over each folder in the processed folder
  for folder in os.listdir(processed_path):
    # Check if actually folder
    if os.path.isdir(os.path.join(processed_path, folder)):
      # Get segmentation image
      segment_image = np.load(
          os.path.join(processed_path, folder, "segmentation.npy")
      )
      # Loop over before and after correction
      for name in representatives.keys():
        # Parse the msi file
        with ImzMLParser(
            os.path.join(processed_path, folder, f"{name}.imzML")
        ) as reader:
          # Get full msi
          mzs, img = read_msi(reader)
          # Get tissue spectra's mean
          representatives[name]["tissue"][folder] = img[segment_image, :].mean(
              axis=0
          )
          # Get background spectra's mean
          representatives[name]["background"][folder] = img[
              ~segment_image, :].mean(axis=0)
  return representatives


def correlation_analysis(processed_path: str):
  #
  spectra_types = ["tissue", "background"]

  # Loop over each folder in the processed folder
  for folder in os.listdir(processed_path):
    for spectra_type in spectra_types:
      get_representative_spectra()

  print(get_representative_spectras(processed_path))
  """
  # Create correlation plot combinations
  # s is section and r is replica
  combinations = [
      ("s", "s", "tissue", "tissue"),
      ("s", "s", "background", "background"),
      ("s", "s", "tissue", "background"),
      ("r", "r", "tissue", "tissue"),
      ("r", "r", "background", "background"),
      ("r", "r", "tissue", "background"),
      ("s", "r", "tissue", "tissue"),
      ("s", "r", "background", "background"),
      ("s", "r", "tissue", "background"),
  ]
  
  for combination in combinations:
    # Calculate correlation before correction
    correlation(processed_path, True, *combination)
    # Calculate correlation after correction
    correlation(processed_path, False, *combination)
  """