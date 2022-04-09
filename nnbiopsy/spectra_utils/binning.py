"""Mass spectrometry spectra binning

This module should be imported and contains the following:

    * BinningInterface - Interface for a spectra binning.
    * EqualWidthBinning - Class for equal width spectra binning.

"""

import numpy as np
from scipy import stats
from typing import Tuple
from abc import ABC, abstractmethod


class BinningInterface(ABC):
  """Interface for a spectra binning.

  """

  @abstractmethod
  def bin(
      self, spectra: Tuple[np.ndarray,
                           np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to bin a spectra

    Args:
        spectra (Tuple[np.ndarray, np.ndarray]): First element is the mz values
        array of spectra and second element is the intensity values array
        of spectra.

    Returns:
        Tuple[np.ndarray, np.ndarray]: First element is the mz values array of
        spectra and second element is the binned intensity values array
        of spectra.

    """
    raise NotImplementedError


class EqualWidthBinning(BinningInterface):
  """Equal width spectra binning.

  """

  def __init__(self, mz_start: float, mz_end: float, bin_width: float) -> None:
    """__init__ method.

    Args:
        mz_start (float): Mz spectra range start.
        mz_end (float): Mz spectra range end.
        bin_width (float): Binning bin width.
    """
    super().__init__()

    # Calculate number of bins
    num_bins = int((mz_end - mz_start) / bin_width)
    # create bin edges array of equal width bins from mz_start to
    # mz_end with bin_width
    self.bin_edges = np.around(np.linspace(mz_start, mz_end, num_bins + 1), 5)
    # create bin centers array of equal width bins from mz_start to
    # mz_end with bin_width
    self.bin_centers = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])

  def bin(
      self, spectra: Tuple[np.ndarray,
                           np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to bin spectra

    Args:
        spectra (Tuple[np.ndarray, np.ndarray]): First element is mz values
        array of spectra and second element is the intensity values array
        of spectra.

    Returns:
        Tuple[np.ndarray, np.ndarray]: First element is mz values array of
        spectra and second element is normalized intensity values array
        of spectra.

    """
    # unpack spectra
    mzs, intensities = np.copy(spectra)

    # if no mz values
    if np.array_equal(mzs, []):
      return self.bin_centers, np.zeros(self.bin_centers.shape)

    # bin data
    bin_sums, _, _ = stats.binned_statistic(x=mzs,
                                            values=intensities,
                                            statistic=sum,
                                            bins=self.bin_edges)

    return self.bin_centers, bin_sums
