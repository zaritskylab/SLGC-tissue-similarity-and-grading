""" This module contain all utils for mass spectrum """

import numpy as np
import pandas as pd
from typing import Tuple


class SpectrumUtils():
  """
  Class that contain all utils for mass spectrum
  """
  def __init__(self) -> None:
    pass

  @classmethod
  def normalize_spectrum(
      cls, spectrum: Tuple[np.ndarray, np.ndarray]
      ) -> Tuple[np.ndarray, np.ndarray]:    
    """
    Method to normalize spectrum intensity by total ion count (TIC).

    Args:
      spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
      array of spectrum and second element is the intensity values array
      of spectrum.

    Returns:
      Tuple[np.ndarray, np.ndarray]: first element is mz values array of
      spectrum and second element is normalized intensity values array
      of spectrum.
    """
    # unpack spectrum
    mzs, intensities = spectrum
    # get TIC - total ion count
    intensities_sum = intensities.sum()
    # if TIC is zero no need to divide by TIC
    if  intensities_sum == 0:
      return (mzs, intensities)
    return  (mzs, intensities / intensities_sum)

  @classmethod
  def bining_spectrum(
      cls, spectrum: Tuple[np.ndarray, np.ndarray],
      bins:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Method to bin spectrum intensity into predefined bins

    Args:
      spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
      array of spectrum and second element is the intensity values array
      of spectrum.
      bins (np.ndarray): predefined mz values bins array.

    Returns:
      Tuple[np.ndarray, np.ndarray]: first element is the predefined mz values
      bins and second element is bin average intensity values array of spectrum.
    """
    # unpack spectrum
    mzs, intensities = spectrum

    # create new empty intensities array
    # corresponding to bins
    new_intensities = np.zeros(bins.shape)

    # assign each mz value to its corresponding bin index
    mz_bin_index = np.digitize(mzs, bins, right=True)

    # create dataframe with the following columns -
    # mz value bin index, mz value, intensity
    df = pd.DataFrame({'mz_bin_index': np.asarray(mz_bin_index),
              'mz': np.asarray(mzs), 'intensity': np.asarray(intensities)})

    # group by the bins index, get the intensity 
    # as sum of bin intensities
    df_group = df.groupby(by='mz_bin_index').agg(
        {'mz_bin_index': 'first', 'intensity': 'sum'})

    # for all bins indexes that are in the mz_bin_index
    # assign the corresponding intensity value
    # leaving all other bins zero
    new_intensities[df_group['mz_bin_index']] = df_group['intensity']

    return (bins.copy(), new_intensities)
