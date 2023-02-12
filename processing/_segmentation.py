"""Mass spectrometry image segmentation
This module should be imported and contains the following:
    * SegmentationInterface - Interface for msi segmentation.
    * MeanSegmentation - Class for msi mean segmentation.

"""

import numpy as np
from typing import List
from abc import ABC, abstractmethod
from skimage import filters
from skimage.morphology import disk


class SegmentationInterface(ABC):
  """Interface for msi segmentation.

  """

  @abstractmethod
  def segment(self, img: np.ndarray) -> np.ndarray:
    """Method to segment msi image.
    
    Args:
        img (np.ndarray): Mass spectrum image.
    
    Returns:
      np.ndarray: Segmentation image.

    """
    raise NotImplementedError


class MeanSegmentation(SegmentationInterface):
  """Mean segmentation for msi

  """

  def __init__(
      self, mzs: np.ndarray, representative_peaks: List[float],
      mass_resolution: float
  ) -> None:
    """__init__ method.

    Args:
        mzs (np.ndarray): Continues mzs values.
        representative_peaks (List[float]): Representative peaks (mz values) 
            for getting a single channel image.
        mass_resolution (float): Mass resolution of the msi.

    """
    self.mzs = mzs
    self.peaks = representative_peaks
    self.res = mass_resolution

  def segment(self, img: np.ndarray) -> np.ndarray:
    """Method to segment msi image.
    
    Args:
        img (np.ndarray): Continues mass spectrum image.
    
    Returns:
      np.ndarray: Segmentation image.

    """
    # Define filter of peaks
    filter_all = False
    for peak in self.peaks:
      filter_all |= (
          (self.mzs >= peak - self.res) & (self.mzs <= peak + self.res)
      )
    # Get peaks accumulative image
    peaks_img = img[:, :, filter_all].sum(axis=-1)
    # Remove salt and pepper noise
    smooth = filters.median(peaks_img, disk(2))
    # Threshold image
    return (smooth > filters.threshold_mean(smooth))
