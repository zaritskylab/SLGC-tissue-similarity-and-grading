"""Mass spectrometry spectral binning

This module should be imported and contains the following mass spectrometry
spectral binning classes:

    * SpectrumBinning - Base class for a spectrum binning.
    * MassResolutionBinning - Mass resolution binning class.

"""

import numpy as np
import pandas as pd
from typing import Tuple
from abc import ABC, abstractmethod


class SpectrumBinning(ABC):
  """Base class for a spectrum binning.

  Each child class has to implement the bin spectrum method

  """

  @abstractmethod
  def bin(
      self, spectrum: Tuple[np.ndarray,
                            np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Abstract Method to bin a spectrum.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.

    Returns:
        Tuple[np.ndarray, np.ndarray]: first element is mz values array of
        spectrum and second element is normalized intensity values array
        of spectrum.

    """
    pass


class MassResolutionBinning(SpectrumBinning):
  """Mass resolution binning class.

  """

  def __init__(self, mz_start: float, mz_end: float,
               mass_resolution: float) -> None:
    """__init__ method.

    Args:
        mz_start (float): mz spectrum range start.
        mz_end (float): mz spectrum range end.
        mass_resolution (float): mass spectrometry resolution.
    """
    super().__init__()

    # create spectrum bins using spectrum lowest and largest
    # mz value and spectrum mass resolution
    self.bins = np.around(np.arange(mz_start, mz_end, mass_resolution / 2), 5)

  def bin(
      self, spectrum: Tuple[np.ndarray,
                            np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Abstract Method to normalize a spectrum total ion count (TIC).

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

    # create new empty intensities array
    # corresponding to bins
    new_intensities = np.zeros(self.bins.shape)

    # assign each mz value to its corresponding bin index
    mz_bin_index = np.digitize(mzs, self.bins, right=True)

    # create dataframe with the following columns -
    # mz value bin index, mz value, intensity
    df = pd.DataFrame({
        'mz_bin_index': np.asarray(mz_bin_index),
        'mz': np.asarray(mzs),
        'intensity': np.asarray(intensities)
    })

    # group by the bins index, get the intensity
    # as sum of bin intensities
    df_group = df.groupby(by='mz_bin_index').agg({
        'mz_bin_index': 'first',
        'intensity': 'sum'
    })

    # for all bins indexes that are in the mz_bin_index
    # assign the corresponding intensity value
    # leaving all other bins zero
    new_intensities[df_group['mz_bin_index']] = df_group['intensity']

    return (self.bins.copy(), new_intensities)
