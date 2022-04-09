"""TIC normalization testing

"""

import numpy as np
from nnbiopsy.spectra_utils import normalization


def test_normalization_intensities_values_are_zero():
  # Arrange
  mzs = np.asarray([10.1, 40.5, 150.0, 1000.3, 140.8, 70.9])
  in_intensities = np.asarray([0, 0, 0, 0, 0, 0])
  out_intensities = np.asarray([0, 0, 0, 0, 0, 0])
  spectra = (mzs, in_intensities)

  # Act
  new_spectra = normalization.TICNormalizer().normalize(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_normalization_intensities_values_are_same():
  # Arrange
  mzs = np.asarray([10.1, 40.5, 150.0, 1000.3, 140.8, 70.9])
  in_intensities = np.asarray([30, 30, 30, 30, 30, 30])
  out_intensities = np.asarray([
      0.16666574074588475, 0.16666574074588475, 0.16666574074588475,
      0.16666574074588475, 0.16666574074588475, 0.16666574074588475
  ])
  spectra = (mzs, in_intensities)

  # Act
  new_spectra = normalization.TICNormalizer().normalize(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_normalization_intensities_values_are_different():
  # Arrange
  mzs = np.asarray([10.1, 40.5, 150.0, 1000.3, 140.8, 70.9])
  in_intensities = np.asarray([5.5, 3.4, 80.5, 30.2, 30.0, 0.0])
  out_intensities = np.asarray([
      0.036764460130614095, 0.022727120808015984, 0.5380980073662608,
      0.20187030835355374, 0.20053341889425869, 0.0
  ])
  spectra = (mzs, in_intensities)

  # Act
  new_spectra = normalization.TICNormalizer().normalize(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_normalization_empty_spectra():
  # Arrange
  mzs = np.asarray([])
  in_intensities = np.asarray([])
  out_intensities = np.asarray([])
  spectra = (mzs, in_intensities)

  # Act
  new_spectra = normalization.TICNormalizer().normalize(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_spectra_not_numpy_array():
  # Arrange
  mzs = []
  in_intensities = []
  out_intensities = []
  spectra = (mzs, in_intensities)

  # Act
  new_spectra = normalization.TICNormalizer().normalize(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True
