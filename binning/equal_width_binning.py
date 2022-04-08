"""Mass spectrometry spectra binning

This module should be imported and contains the following:

    * EqualWidthBinning - Class for equal width spectra binning.

"""

import numpy as np
from scipy import stats
from typing import Tuple
from NanoBiopsy.binning.binning_interface import BinningInterface


class EqualWidthBinning(BinningInterface):
  """Equal width spectra binning.

  """

  def __init__(self, mz_start: float, mz_end: float,
               bin_width: float) -> None:
    """__init__ method.

    Args:
        mz_start (float): mz spectrum range start.
        mz_end (float): mz spectrum range end.
        bin_width (float): Binning bin width.
    """
    super().__init__()

    # Calculate number of bins
    num_bins = int((mz_end - mz_start) / bin_width)
    # create bin edges array of equal width bins from mz_start to mz_end with bin_width
    self.bin_edges = np.around(np.linspace(mz_start, mz_end, num_bins), 5)
    # create bin centers array of equal width bins from mz_start to mz_end with bin_width 
    self.bin_centers = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])

  def bin(
      self, spectrum: Tuple[np.ndarray,
                            np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to bin spectra

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
    
    # bin data
    bin_sums, _, _ = stats.binned_statistic(x=mzs, values=intensities, statistic=sum, bins=self.bin_edges)
    
    return self.bin_centers, bin_sums
