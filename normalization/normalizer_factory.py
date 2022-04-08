"""Mass spectrometry spectra normalization factory

This module should be imported and contains the following:

    * NormalizerFactory - Factory to create spectra normalizer object.

"""

from NanoBiopsy.normalization.normalizer_interface import NormalizerInterface
from NanoBiopsy.normalization.tic_normalizer import TICNormalizer
from NanoBiopsy.normalization.median_normalizer import MedianNormalizer


class NormalizerFactory():
  """Normalizer Factory

  """

  @classmethod
  def get_normalizer(cls, n_type='TIC') -> NormalizerInterface:
    """Method to get normalizer by normalizer string type.

    Args:
        n_type (str, optional): Normalier type can be one of the following
        ['TIC', 'Median']. Defaults to 'TIC'.

    Raises:
        ValueError: if normalier string type does not correspond to a valid
        n_type.

    Returns:
        NormalizerInterface: normalizer interface.
    """
    if n_type == 'TIC':
      return TICNormalizer()
    elif n_type == 'Median':
      return MedianNormalizer()
    else:
      raise ValueError(n_type)
