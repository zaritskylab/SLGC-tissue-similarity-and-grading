"""Mass spectrometry spectra binning interface

This module should be imported and contains the following:

    * BinningInterface - Interface for a spectra binning.

"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class BinningInterface(ABC):
  """Base class for a spectra binning.

  """

  @abstractmethod
  def bin(
      self, spectra: Tuple[np.ndarray,
                           np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to bin a spectra

    Args:
        spectra (Tuple[np.ndarray, np.ndarray]): first element is the mz values
        array of spectra and second element is the intensity values array
        of spectra.

    Returns:
        Tuple[np.ndarray, np.ndarray]: first element is the mz values array of
        spectra and second element is the binned intensity values array
        of spectra.

    """
    raise NotImplementedError
