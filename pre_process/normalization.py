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
  def normalize(cls,
                spectrum: Tuple[np.ndarray, np.ndarray],
                epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Abstract Method to normalize a spectrum.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]: first element is mz values array of
        spectrum and second element is normalized intensity values array
        of spectrum.

    """
    pass

  @classmethod
  @abstractmethod
  def region_normalize(cls,
                       spectrum: Tuple[np.ndarray, np.ndarray],
                       regions: List[Tuple[int, int]],
                       epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Abstract Method to region normalize a spectrum.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        regions (List[Tuple[int, int]]): list of regions to apply normalization
        to. Each element should have a region start value and end value.
        assumptions is there is no intersection between regions.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

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
  def normalize(cls,
                spectrum: Tuple[np.ndarray, np.ndarray],
                epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectrum by total ion count.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]: first element is mz values array of
        spectrum and second element is normalized intensity values array
        of spectrum.

    """
    # unpack spectrum
    mzs, intensities = np.copy(spectrum)
    # return tic normalized
    return (mzs, intensities / (intensities.sum() + epsilon))

  @classmethod
  def region_normalize(cls,
                       spectrum: Tuple[np.ndarray, np.ndarray],
                       regions: List[Tuple[int, int]],
                       epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Method to region normalize a spectrum by total ion count.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        regions (List[Tuple[int, int]]): list of regions to apply normalization
        to. Each element should have a region start value and end value. 
        assumptions is there is no intersection between regions.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

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
      # apply tic normalization
      intensities[idx] = (intensities[idx] / (intensities[idx].sum() + epsilon))
    return (mzs, intensities)


class MedianNormalizer(SpectrumNormalizer):
  """Median normalizer.

  """

  @classmethod
  def normalize(cls,
                spectrum: Tuple[np.ndarray, np.ndarray],
                epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectrum by median.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]: first element is mz values array of
        spectrum and second element is normalized intensity values array
        of spectrum.

    """
    # unpack spectrum
    mzs, intensities = np.copy(spectrum)
    # return median normalized
    return (mzs, intensities / (np.median(intensities) + epsilon))

  @classmethod
  def region_normalize(cls,
                       spectrum: Tuple[np.ndarray, np.ndarray],
                       regions: List[Tuple[int, int]],
                       epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Method to region normalize a spectrum by median.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        regions (List[Tuple[int, int]]): list of regions to apply normalization
        to. Each element should have a region start value and end value. 
        assumptions is there is no intersection between regions.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

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
      # apply median normalization
      intensities[idx] = (intensities[idx] /
                          (np.median(intensities[idx]) + epsilon))
    return (mzs, intensities)


class MeanNormalizer(SpectrumNormalizer):
  """Mean normalizer.

  """

  @classmethod
  def normalize(cls,
                spectrum: Tuple[np.ndarray, np.ndarray],
                epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectrum by mean.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]: first element is mz values array of
        spectrum and second element is normalized intensity values array
        of spectrum.

    """
    # unpack spectrum
    mzs, intensities = np.copy(spectrum)
    # return mean normalized
    return (mzs, intensities / (np.mean(intensities) + epsilon))

  @classmethod
  def region_normalize(cls,
                       spectrum: Tuple[np.ndarray, np.ndarray],
                       regions: List[Tuple[int, int]],
                       epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Method to region normalize a spectrum by mean.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        regions (List[Tuple[int, int]]): list of regions to apply normalization
        to. Each element should have a region start value and end value. 
        assumptions is there is no intersection between regions.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

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
      # apply mean normalization
      intensities[idx] = (intensities[idx] /
                          (np.mean(intensities[idx]) + epsilon))
    return (mzs, intensities)


class Q3Normalizer(SpectrumNormalizer):
  """Q3 normalizer.

  """

  @classmethod
  def normalize(cls,
                spectrum: Tuple[np.ndarray, np.ndarray],
                epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Method to normalize a spectrum by Q3.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

    Returns:
        Tuple[np.ndarray, np.ndarray]: first element is mz values array of
        spectrum and second element is normalized intensity values array
        of spectrum.

    """
    # unpack spectrum
    mzs, intensities = np.copy(spectrum)
    # return mean normalized
    return (mzs, intensities / (np.quantile(intensities, 0.75) + epsilon))

  @classmethod
  def region_normalize(cls,
                       spectrum: Tuple[np.ndarray, np.ndarray],
                       regions: List[Tuple[int, int]],
                       epsilon: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """Method to region normalize a spectrum by Q3.

    Args:
        spectrum (Tuple[np.ndarray, np.ndarray]): first element is mz values
        array of spectrum and second element is the intensity values array
        of spectrum.
        regions (List[Tuple[int, int]]): list of regions to apply normalization
        to. Each element should have a region start value and end value. 
        assumptions is there is no intersection between regions.
        epsilon (float): small float added to denominator  to avoid dividing
        by zero.

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
      # apply mean normalization
      intensities[idx] = (intensities[idx] /
                          (np.quantile(intensities[idx], 0.75) + epsilon))
    return (mzs, intensities)


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
