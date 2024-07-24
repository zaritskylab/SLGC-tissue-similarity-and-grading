"""Mass spectrometry spectra normalization
This module should be imported and contains the following:
    
    * NormalizerInterface - Interface for a spectra normalizer.
    * TICNormalizer - Class for total ion count normalization.

"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class LockMassInterface(ABC):
    """Interface for a spectra lock mass

  """

    @classmethod
    @abstractmethod
    def lock_mass(
        cls, spectra: Tuple[np.ndarray,
                            np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Method to lock mass a spectra
    Args:
        spectra (Tuple[np.ndarray, np.ndarray]): First element is the mz values
        array of spectra and second element is the intensity values array
        of spectra.
    Returns:
        Tuple[np.ndarray, np.ndarray]: First element is the mz values array of
        spectra after lock mass and second element is the intensity values array
        of spectra.

    """
        raise NotImplementedError


class ReferenceLockMass(LockMassInterface):
    """Lock mass spectra by reference peak

  """

    def __init__(self,
                 original_lock_mass_position: float,
                 representative_spectra: Tuple[np.ndarray, np.ndarray],
                 tol: float = 0.3) -> None:
        """__init__ method.

    Args:
        original_lock_mass_position (float): The original peak value for expected.
        representative_spectra (Tuple[np.ndarray, np.ndarray]): Representative
            spectra to find the shifted peak we expect, suggested is mean spectra.
            first element is the mz values array of spectra and second element 
            is the intensity values array of spectra.
        tol (float, optional): Tolerance for searching the shifted peak from expected peak.
            Defaults to 0.3.

    """
        super().__init__()

        # Unpack representative spectra
        mzs, intensities = representative_spectra
        # Filter to search mzs in reference peak +- tol
        mzs_filter = ((mzs >= original_lock_mass_position - tol) &
                      (mzs <= original_lock_mass_position + tol))
        # Find peak with max intensity in reference peak +- tol
        shifted_index = np.argmax(intensities[mzs_filter])
        # Get shifted peak mz value
        shifted_lock_mass_position = mzs[mzs_filter][shifted_index]
        # Store scale ration for lock mass
        self.scale_ratio = (original_lock_mass_position /
                            shifted_lock_mass_position)
        #
        self.diff = original_lock_mass_position - shifted_lock_mass_position
        print(self.scale_ratio, self.diff)

    def lock_mass(
        self, spectra: Tuple[np.ndarray,
                             np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Method to lock mass a spectra
    Args:
        spectra (Tuple[np.ndarray, np.ndarray]): First element is the mz values
        array of spectra and second element is the intensity values array
        of spectra.
    Returns:
        Tuple[np.ndarray, np.ndarray]: First element is the mz values array of
        spectra after lock mass and second element is the intensity values array
        of spectra.

    """
        # Unpack spectra
        mzs, intensities = np.copy(spectra)
        # Scale each mz value
        scaled_mzs = [self.scale_ratio * mz for mz in mzs]
        # scaled_mzs = [mz + self.diff for mz in mzs]
        return (scaled_mzs, intensities)
