"""Mass spectrometry spectra median normalization

This module should be imported and contains the following:

    * MedianNormalizer - Class for median normalization.

"""

import numpy as np
from typing import Tuple
from normalization.normalizer_interface import NormalizerInterface


class MedianNormalizer(NormalizerInterface):
  """Median normalizer.

  """

  @classmethod
  def normalize(cls,
                spectra: Tuple[np.ndarray, np.ndarray],
                epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectra

    Args:
        spectra (Tuple[np.ndarray, np.ndarray]): first element is the mz values
        array of spectra and second element is the intensity values array
        of spectra.
        epsilon (float): small float added to denominator to avoid zero
        division.

    Returns:
        Tuple[np.ndarray, np.ndarray]: first element is the mz values array of
        spectra and second element is the normalized intensity values array
        of spectra.

    """
    # unpack spectra
    mzs, intensities = np.copy(spectra)
    # return median normalized
    return (mzs, intensities / (np.median(intensities) + epsilon))
