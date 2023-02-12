"""Mass spectrometry image correction
This module should be imported and contains the following:
    * CorrectionInterface - Interface for msi correction.
    * ZScoreCorrection - Class for z-score correction.

"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler


class CorrectionInterface(ABC):
  """Interface for msi correction.

  """

  @abstractmethod
  def correct(self, img: np.ndarray) -> np.ndarray:
    """Method to correct msi image.
    
    Args:
        img (np.ndarray): Mass spectrum image.
    
    Returns:
      np.ndarray: Corrected image.

    """
    raise NotImplementedError


class ZScoreCorrection(CorrectionInterface):

  def correct(self, img: np.ndarray, segment_img: np.ndarray) -> np.ndarray:
    """Method to correct msi image using zscore calculated from background
        spectras.
    
    Args:
        img (np.ndarray): Mass spectrum image.
        segment_img (np.ndarray): Segmentation image.
    
    Returns:
      np.ndarray: Corrected image.

    """
    # Create z-score object using background spectras
    scalar = StandardScaler().fit(img[~segment_img, :])
    # Reshape image to 2d array
    zscore_data = img.copy().reshape(
        (img.shape[0] * img.shape[1], img.shape[2])
    )
    # Transform all spectras
    zscore_data = scalar.transform(zscore_data)
    # Reshape image to original shape
    zscore_data = zscore_data.reshape(
        (img.shape[0], img.shape[1], img.shape[2])
    )
    return zscore_data
