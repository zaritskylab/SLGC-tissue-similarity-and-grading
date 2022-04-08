"""Mass spectrometry spectra tic normalization

This module should be imported and contains the following:

    * TICNormalizer - Class for total ion count normalization.

"""

import numpy as np
from typing import Tuple
from NanoBiopsy.normalization.normalizer_interface import NormalizerInterface


class TICNormalizer(NormalizerInterface):
  """Total Ion Count normalizer.

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
    # return tic normalized
    return (mzs, intensities / (intensities.sum() + epsilon))
