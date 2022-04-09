from os import listdir
from os.path import join
from pathlib import Path
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter
from matplotlib import pyplot as plt
from tqdm import tqdm
from nnbiopsy.spectra_utils.normalization import TICNormalizer
from nnbiopsy.spectra_utils.binning import EqualWidthBinning

# Define folder that contains the dhg dataset
DHG_IN_PATH = "/sise/assafzar-group/assafzar/Leor/DHG/Original"
# Define folder to save preprocess output
DHG_OUT_PATH = "/sise/assafzar-group/assafzar/Leor/DHG/Normalized"
# Define mass range start value
MZ_START = 50
# Define mass range end value
MZ_END = 1200
# # Define mass resolution of the data
MASS_RESOLUTION = 0.025

# Create DHG_OUT_PATH folder if doesn"t exist
Path(DHG_OUT_PATH).mkdir(parents=True, exist_ok=True)

# Get all msi names
msi_names = [file for file in listdir(DHG_IN_PATH) if file.endswith(".imzML")]

# Get normalizer object
normalizer = TICNormalizer()
# Get binning object
binning = EqualWidthBinning(MZ_START, MZ_END, MASS_RESOLUTION / 2)

# Create preprocess pipe
pre_process_pipe = (lambda mzs, intensities:
                    (binning.bin(normalizer.normalize((mzs, intensities)))))

# Loop over each msi name
for msi_name in tqdm(msi_names, desc="Image Loop"):
  # Create a new preprocessed msi. because we apply binning
  # we can use mode="continuous"
  with ImzMLWriter(join(DHG_OUT_PATH, msi_name), mode="continuous") as writer:
    # Parse the msi file
    with ImzMLParser(join(DHG_IN_PATH, msi_name)) as reader:
      # Loop over each spectra in msi
      for idx, (x, y, z) in tqdm(enumerate(reader.coordinates),
                                 total=len(reader.coordinates),
                                 desc="Spectra Loop"):
        # Read spectra
        raw_mzs, raw_intensities = reader.getspectrum(idx)
        # Apply preprocessing pipe
        preprocessed_mzs, preprocessed_intensities = pre_process_pipe(
            raw_mzs, raw_intensities)
        # Write processed spectra to new preprocessed msi
        writer.addSpectrum(preprocessed_mzs, preprocessed_intensities,
                           (x, y, z))
