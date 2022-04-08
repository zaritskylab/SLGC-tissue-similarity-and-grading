"""Mass spectrometry spectra binning factory

This module should be imported and contains the following:

    * BinningFactory - Factory to create spectra binning object.

"""

from binning.binning_interface import BinningInterface
from binning.equal_width_binning import EqualWidthBinning


class BinningFactory():
  """Binning Factory

  """

  @classmethod
  def get_binning(cls, n_type='Equal_Width') -> BinningInterface:
    """Method to get binning by binning string type.

    Args:
        n_type (str, optional): Binning type can be one of the following
        ['Equal_Width']. Defaults to 'Equal_Width'.

    Raises:
        ValueError: if binning string type does not correspond to a valid
        n_type.

    Returns:
        BinningInterface: binning interface.
    """
    if n_type == 'Equal_Width':
      return EqualWidthBinning()
    else:
      raise ValueError(n_type)
