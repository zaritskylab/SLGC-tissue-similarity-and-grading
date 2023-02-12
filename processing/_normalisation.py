"""Mass spectrometry spectra normalization
This module should be imported and contains the following:
    
    * NormalizerInterface - Interface for a spectra normalizer.
    * TICNormalizer - Class for total ion count normalization.

"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class NormalizerInterface(ABC):
  """Interface for a spectra normalizer

  """

  @classmethod
  @abstractmethod
  def normalize(
      cls, spectra: Tuple[np.ndarray, np.ndarray], epsilon: float = 0.001
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectra
    Args:
        spectra (Tuple[np.ndarray, np.ndarray]): First element is the mz values
        array of spectra and second element is the intensity values array
        of spectra.
        epsilon (float): Small float added to denominator to avoid zero
        division.
    Returns:
        Tuple[np.ndarray, np.ndarray]: First element is the mz values array of
        spectra and second element is the normalized intensity values array
        of spectra.

    """
    raise NotImplementedError


class TICNormalizer(NormalizerInterface):
  """Total Ion Count normalizer.

  """

  @classmethod
  def normalize(
      cls, spectra: Tuple[np.ndarray, np.ndarray], epsilon: float = 0.001
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectra

    Args:
        spectra (Tuple[np.ndarray, np.ndarray]): First element is the mz values
        array of spectra and second element is the intensity values array
        of spectra.
        epsilon (float): Small float added to denominator to avoid zero
        division.

    Returns:
        Tuple[np.ndarray, np.ndarray]: First element is the mz values array of
        spectra and second element is the normalized intensity values array
        of spectra.

    """
    # Unpack spectra
    mzs, intensities = np.copy(spectra)
    # Return tic normalized
    return (mzs, intensities / (intensities.sum() + epsilon))
