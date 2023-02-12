"""Mass spectrometry spectra discretization
This module should be imported and contains the following:
    * BinningInterface - Interface for a spectra binning.
    * EqualWidthBinning - Class for equal width spectra binning.

"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class BinningInterface(ABC):
  """Interface for spectra binning.

  """

  @abstractmethod
  def bin(
      self, spectra: Tuple[np.ndarray, np.ndarray]
  ) -> Tuple[np.ndarray, np.ndarray]:
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
    # Create bin edges array of equal width bins from mz_start to
    # mz_end with bin_width
    self.bin_edges = np.around(np.linspace(mz_start, mz_end, num_bins + 1), 5)
    # Create bin centers array of equal width bins from mz_start to
    # mz_end with bin_width
    self.bin_centers = 0.5 * (self.bin_edges[1:] + self.bin_edges[:-1])

  def bin(
      self, spectra: Tuple[np.ndarray, np.ndarray]
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Method to bin spectra

    Args:
        spectra (Tuple[np.ndarray, np.ndarray]): First element is mz values
        array of spectra and second element is the intensity values array
        of spectra.

    Returns:
        Tuple[np.ndarray, np.ndarray]: First element is the mz values array of
        spectra and second element is the binned intensity values array
        of spectra.

    """
    # Unpack spectra
    mzs, intensities = np.copy(spectra)

    # If no mz values
    if np.array_equal(mzs, []):
      return self.bin_centers, np.zeros(self.bin_centers.shape)

    # Assign each mz value to its corresponding bin index
    bin_index = np.digitize(mzs, self.bin_edges)

    # Make sure index of spectra shape appears in bin_index
    # for shape consistency
    bin_index = np.append(bin_index, self.bin_centers.shape[0])
    intensities = np.append(intensities, 0)

    # Assign values beyond the bounds of bins to first and last bin
    # if they are smaller\larger than bounds respectively
    bin_index[bin_index == 0] = 1
    bin_index[bin_index == len(self.bin_edges)] = len(self.bin_edges) - 1

    # 0 index
    bin_index -= 1

    # Sum value of each bins
    bin_sums = np.bincount(bin_index, weights=intensities)

    return self.bin_centers, bin_sums
