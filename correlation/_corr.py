"""Nanobiopsy correlation analysis
This module should be imported and contains the following:
    
    * _correlation - Function to calculate correlation matrix
    * _get_representative_spectras - Function to get a representative spectra
        from each biopsy before and after correction.
    * correlation_analysis - Function to apply a correlation analysis.

"""

import os
import itertools
import numpy as np
import pandas as pd
from typing import List, Dict
from scipy import stats
from pyimzml.ImzMLParser import ImzMLParser
from utils import read_msi


def _correlation(
    i_keys: List[str], j_keys: List[str], i_vals: Dict[str, np.ndarray],
    j_vals: Dict[str, np.ndarray], corr_type: str = "pearson"
) -> pd.DataFrame:
  """Function to calculate correlation matrix.

  Args:
      i_keys (List[str]):  Matrix i (rows) keys.
      j_keys (List[str]): Matrix i (columns) keys.
      i_vals (Dict[str, np.ndarray]): Dict with keys corresponding to i_keys
        and values of the keys.
      j_vals (Dict[str, np.ndarray]): Dict with keys corresponding to j_keys
        and values of the keys.
      corr_type (str, optional): Correlation type . Defaults to "pearson".
      

  Returns:
      pd.DataFrame: Correlation matrix where index is i_keys and columns are
        j_keys and each cell [i_key, j_key] is the correlation between i_val
        and j_val.

  """
  # Create empty correlation matrix
  corr_m = np.zeros((len(i_keys), len(j_keys)))
  # Loop over i keys
  for idx_i, key_i in enumerate(i_keys):
    # Loop over j keys
    for idx_j, key_j in enumerate(j_keys):
      # Calculate corelation between i_val and j_val
      if corr_type == "kendall":
        corr_m[idx_i, idx_j] = stats.kendalltau(i_vals[key_i], j_vals[key_j])[0]
      elif corr_type == "spearman":
        corr_m[idx_i, idx_j] = stats.spearmanr(i_vals[key_i], j_vals[key_j])[0]
      else:
        corr_m[idx_i, idx_j] = stats.pearsonr(i_vals[key_i], j_vals[key_j])[0]
  # return correlation data frame
  return pd.DataFrame(corr_m, index=i_keys, columns=j_keys)


def _get_representative_spectras(
    processed_path: str
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
  """Function to get a representative spectra from each biopsy before and 
      after correction.

  Args:
      processed_path (str): Path to processed continuos imzML files.
  
  Returns:
      Dict[str, Dict[str, Dict[str, np.ndarray]]]: Representative spectra from
          each biopsy before and after correction. Contains dictionaries 
          'common_representation' and 'meaningful_signal' and for each of the
          dictionary there are 'tissue' and 'background' dictionaries which 
          contain biopsies names and their representative spectra.

  """
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


def correlation_analysis(processed_path: str, output_path: str) -> None:
  """Function to apply a correlation analysis.

  Args:
      processed_path (str): Path to processed continuos imzML files. 
      output_path (str): Path to save correlation matrices.
  """
  # Get representative spectra from each biopsy before and after correction
  representative_spectras = _get_representative_spectras(processed_path)

  # Loop over before and after correction
  for image_type in representative_spectras.keys():
    # Loop over combinations of tissue and background
    for combination in list(
        itertools.combinations_with_replacement(
            representative_spectras[image_type].keys(), 2
        )
    ):
      # Get correlation matrix
      corr_df = _correlation(
          list(representative_spectras[image_type][combination[0]].keys()),
          list(representative_spectras[image_type][combination[1]].keys()),
          representative_spectras[image_type][combination[0]],
          representative_spectras[image_type][combination[1]]
      )
      # Save to csv
      corr_df.to_csv(
          os.path.join(
              output_path,
              image_type.replace("_", "-") + "_" + "_".join(combination) +
              ".csv"
          )
      )
