"""Mass spectrometry spectral normalization

This module should be imported and contains the following mass spectrometry
spectral normalizers:

    * SpectrumNormalizer - Base class for a spectrum normalizer.
    * TICNormalizer - Total Ion Count normalizer.

"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class SpectrumNormalizer(ABC):
  """Base class for a spectrum normalizer.

  Each child class has to implement the normalize spectrum method

  """

  @classmethod
  @abstractmethod
  def normalize(
      cls, spectrum: Tuple[np.ndarray,
                           np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Abstract Method to normalize a spectrum.

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


class TICNormalizer(SpectrumNormalizer):
  """Total Ion Count normalizer.

  """

  @classmethod
  def normalize(
      cls, spectrum: Tuple[np.ndarray,
                           np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Abstract Method to normalize a spectrum by total ion count (TIC).

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
    if intensities_sum == 0:
      return (mzs, intensities)
    return (mzs, intensities / intensities_sum)


class MedianNormalizer(SpectrumNormalizer):
  pass


class Q3Normalizer(SpectrumNormalizer):
  pass
