"""Median normalization testing

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
  new_spectra = normalization.MedianNormalizer().normalize(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_normalization_intensities_values_are_same():
  # Arrange
  mzs = np.asarray([10.1, 40.5, 150.0, 1000.3, 140.8, 70.9])
  in_intensities = np.asarray([30, 30, 30, 30, 30, 30])
  out_intensities = np.asarray([
      0.9999666677777407, 0.9999666677777407, 0.9999666677777407,
      0.9999666677777407, 0.9999666677777407, 0.9999666677777407
  ])
  spectra = (mzs, in_intensities)

  # Act
  new_spectra = normalization.MedianNormalizer().normalize(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_normalization_intensities_values_are_different():
  # Arrange
  mzs = np.asarray([10.1, 40.5, 150.0, 1000.3, 140.8, 70.9])
  in_intensities = np.asarray([5.5, 3.4, 80.5, 30.2, 30.0, 0.0])
  out_intensities = np.asarray([
      0.3098416990592079, 0.1915385048729649, 4.534955777139316,
      1.7013126021069234, 1.690045631232043, 0.0
  ])
  spectra = (mzs, in_intensities)

  # Act
  new_spectra = normalization.MedianNormalizer().normalize(spectra)

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
  new_spectra = normalization.MedianNormalizer().normalize(spectra)

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
  new_spectra = normalization.MedianNormalizer().normalize(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True
