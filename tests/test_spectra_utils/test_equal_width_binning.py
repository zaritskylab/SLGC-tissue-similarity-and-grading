"""Equal width binning testing

"""

import numpy as np
from nnbiopsy.spectra_utils import binning


def test_binning_intensities_values_are_zero():
  # Arrange
  mz_start = 0
  mz_end = 10
  bin_width = 0.5
  in_mzs = np.asarray([0, 4.5, 3.2, 5.7, 9.9, 10])
  in_intensities = np.asarray([0, 0, 0, 0, 0, 0])
  out_mzs = np.asarray([
      0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75,
      6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75
  ])
  out_intensities = np.asarray([
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0
  ])
  spectra = (in_mzs, in_intensities)

  # Act
  new_spectra = binning.EqualWidthBinning(mz_start, mz_end,
                                          bin_width).bin(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], out_mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_binning_intensities_values_are_same():
  # Arrange
  mz_start = 0
  mz_end = 10
  bin_width = 0.5
  in_mzs = np.asarray([0, 4.5, 3.2, 5.7, 9.9, 10])
  in_intensities = np.asarray([30, 30, 30, 30, 30, 30])
  out_mzs = np.asarray([
      0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75,
      6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75
  ])
  out_intensities = np.asarray([
      30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 30.0, 0.0, 30.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 60.0
  ])
  spectra = (in_mzs, in_intensities)

  # Act
  new_spectra = binning.EqualWidthBinning(mz_start, mz_end,
                                          bin_width).bin(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], out_mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_binning_intensities_values_are_different():
  # Arrange
  mz_start = 0
  mz_end = 10
  bin_width = 0.5
  in_mzs = np.asarray([0, 4.5, 3.2, 5.7, 9.9, 10])
  in_intensities = np.asarray([30, 20, 50, 200, 10, 10])
  out_mzs = np.asarray([
      0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75,
      6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75
  ])
  out_intensities = np.asarray([
      30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 20.0, 0.0, 200.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 20.0
  ])
  spectra = (in_mzs, in_intensities)

  # Act
  new_spectra = binning.EqualWidthBinning(mz_start, mz_end,
                                          bin_width).bin(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], out_mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_binning_empty_spectra():
  # Arrange
  mz_start = 0
  mz_end = 10
  bin_width = 0.5
  in_mzs = np.asarray([])
  in_intensities = np.asarray([])
  out_mzs = np.asarray([
      0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75,
      6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75
  ])
  out_intensities = np.asarray([
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0
  ])
  spectra = (in_mzs, in_intensities)

  # Act
  new_spectra = binning.EqualWidthBinning(mz_start, mz_end,
                                          bin_width).bin(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], out_mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True


def test_spectra_not_numpy_array():
  # Arrange
  mz_start = 0
  mz_end = 10
  bin_width = 0.5
  in_mzs = []
  in_intensities = []
  out_mzs = np.asarray([
      0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75,
      6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75
  ])
  out_intensities = np.asarray([
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0
  ])
  spectra = (in_mzs, in_intensities)

  # Act
  new_spectra = binning.EqualWidthBinning(mz_start, mz_end,
                                          bin_width).bin(spectra)

  # Assert
  assert np.array_equal(new_spectra[0], out_mzs) == True
  assert np.array_equal(new_spectra[1], out_intensities) == True
