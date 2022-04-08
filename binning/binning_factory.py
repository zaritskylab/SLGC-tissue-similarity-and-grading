"""Mass spectrometry spectra binning factory

This module should be imported and contains the following:

    * BinningFactory - Factory to create spectra binning object.

"""

from NanoBiopsy.binning.binning_interface import BinningInterface
from NanoBiopsy.binning.equal_width_binning import EqualWidthBinning


class BinningFactory():
  """Binning Factory

  """

  @classmethod
  def get_binning(cls, n_type='Equal_Width', mz_start: float = None,
				  mz_end: float = None, bin_width: float = None) -> BinningInterface:
    """Method to get binning by binning string type.

    Args:
        n_type (str, optional): Binning type can be one of the following
        ['Equal_Width']. Defaults to 'Equal_Width'.
		mz_start (float): Mz spectrum range start.
        mz_end (float): Mz spectrum range end.
        bin_width (float): Binning bin width.

    Raises:
        ValueError: if arguments are incorrect.

    Returns:
        BinningInterface: binning interface.
    """
    if n_type == 'Equal_Width':
      if mz_start is None:
        raise TypeError('mz_start required for equal width binning')
      elif mz_end is None:
        raise TypeError('mz_end required for equal width binning')
      elif bin_width is None:
        raise TypeError('bin_width required for equal width binning')
      return EqualWidthBinning(mz_start, mz_end, bin_width)
      
    else:
      raise ValueError(n_type)
