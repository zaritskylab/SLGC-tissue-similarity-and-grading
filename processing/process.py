"""Mass spectrometry processing
This module should be imported and contains the following:
    
    * process_msi - Function to process msi.
"""

from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from processing.binning import EqualWidthBinning
from processing.normalisation import TICNormalizer


def process_msi(
    input_path: str, output_path: str, x_min: int, x_max: int, y_min: int,
    y_max: int, mz_start: int, mz_end: int, mass_resolution: float
) -> None:
  """Function to process an msi file and create a processed msi file.

  Args:
      input_path (str): Path to imzML file that needs to be processed.
      output_path (str): Path to imzML processed file that will be saved.
      x_min (int): X minimum coordinate of the the tissue in the input.
      x_max (int): X maximum coordinate of the the tissue in the input.
      y_min (int): Y minimum coordinate of the the tissue in the input.
      y_max (int): Y maximum coordinate of the the tissue in the input.
      mz_start (int): The start value of the mz range.
      mz_end (int): The end value of the mz range.
      mass_resolution (float): The mass resolution.
  """
  # Get normalizer object
  normalizer = TICNormalizer()
  # Get binning object
  binning = EqualWidthBinning(mz_start, mz_end, mass_resolution)
  # Create process pipe
  process_pipe = (
      lambda mzs, intensities:
      (binning.bin(normalizer.normalize((mzs, intensities))))
  )
  # Parse the MSI file containing ROI
  with ImzMLParser(input_path) as reader:
    # Create a new MSI for ROI. because we apply binning
    # we can use mode="continuous"
    with ImzMLWriter(output_path, mode="continuous") as writer:
      # Loop over each spectra in MSI
      for idx, (x, y, z) in enumerate(reader.coordinates):
        # Check if spectra is in ROI boundaries
        if ((x_min <= x - 1 <= x_max) & (y_min <= y - 1 <= y_max)):
          # Read spectra from MSI
          raw_mzs, raw_intensities = reader.getspectrum(idx)
          # Apply processing pipe
          preprocessed_mzs, preprocessed_intensities = process_pipe(
              raw_mzs, raw_intensities
          )
          # Write spectra to new MSI with relative coordinate
          writer.addSpectrum(
              preprocessed_mzs, preprocessed_intensities,
              (x - x_min + 1, y - y_min + 1, z)
          )
