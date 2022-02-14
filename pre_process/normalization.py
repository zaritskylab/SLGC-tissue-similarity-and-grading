"""Mass spectrometry spectral normalization

This module should be imported and contains the following mass spectrometry
spectral normalizers:

    * SpectrumNormalizer - Base class for a spectrum normalizer.
    * NormalizerFactory - Factory to create normalizer.
    * TICNormalizer - Total Ion Count normalizer.
    * MedianNormalizer - Median normalizer.
    * MeanNormalizer -Mean normalizer.
    * Q3Normalizer - Q3 normalizer.

"""

import numpy as np
from typing import Tuple, List
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

  @classmethod
  @abstractmethod
  def region_normalize(
      cls, spectrum: Tuple[np.ndarray,
                           np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Abstract Method to region normalize a spectrum.

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
    """Method to normalize a spectrum by total ion count (TIC).

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
    mzs, intensities = np.copy(spectrum)
    # get TIC
    tic = intensities.sum()
    # if TIC is zero no need to divide by TIC
    # all elements are 0
    if tic == 0:
      return (mzs, intensities)
    return (mzs, intensities / tic)

  @classmethod
  def region_normalize(
      cls, spectrum: Tuple[np.ndarray, np.ndarray],
      regions: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to region normalize a spectrum by total ion count (TIC).

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
    mzs, intensities = np.copy(spectrum)
    # loop over each region to apply TIC normalization for that region
    for region in regions:
      # get only region mzs indexes
      idx = (mzs >= region[0] & mzs < region[1])
      # get region TIC
      tic = intensities[idx].sum()
      # if TIC is zero no need to divide by TIC
      # all elements are o
      if tic != 0:
        intensities[idx] = intensities[idx] / tic
    return (mzs, intensities)


class MedianNormalizer(SpectrumNormalizer):
  """Median normalizer.

  """

  @classmethod
  def normalize(
      cls, spectrum: Tuple[np.ndarray,
                           np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectrum by median.

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

  @classmethod
  def region_normalize(
      cls, spectrum: Tuple[np.ndarray, np.ndarray],
      regions: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to region normalize a spectrum by median.

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


class MeanNormalizer(SpectrumNormalizer):
  """Q3 normalizer.

  """

  @classmethod
  def normalize(
      cls, spectrum: Tuple[np.ndarray,
                           np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectrum by mean.

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

  @classmethod
  def region_normalize(
      cls, spectrum: Tuple[np.ndarray, np.ndarray],
      regions: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to region normalize a spectrum by mean.

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


class Q3Normalizer(SpectrumNormalizer):
  """Q3 normalizer.

  """

  @classmethod
  def normalize(
      cls, spectrum: Tuple[np.ndarray,
                           np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectrum by Q3.

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

  @classmethod
  def region_normalize(
      cls, spectrum: Tuple[np.ndarray, np.ndarray],
      regions: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """Method to region normalize a spectrum by Q3.

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


class NormalizerFactory():
  """Normalizer Factory
  
  """

  @classmethod
  def get_normalizer(cls, n_type='TIC') -> SpectrumNormalizer:
    """Method to get normalizer by string type.

    Args:
        type (str, optional): Normalier type can be one of the following
        ['TIC', 'Median', 'Mean', 'Q3']. Defaults to 'TIC'.

    Raises:
        ValueError: if Normalier type is incorrect.

    Returns:
        SpectrumNormalizer: normalizer in its abstract class.
    """
    if n_type == 'TIC':
      return TICNormalizer()
    elif n_type == 'Median':
      return MedianNormalizer()
    elif n_type == 'Mean':
      return MeanNormalizer()
    elif n_type == 'Q3':
      return Q3Normalizer()
    else:
      raise ValueError(n_type)
