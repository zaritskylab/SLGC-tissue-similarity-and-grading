""" Module to create image of mz length distribution plot of all imzML files"""

import argparse
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pyimzml.ImzMLParser import ImzMLParser

if __name__ == "__main__":
  # get command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-i",
                      required=True,
                      help="path to folder containing imzML files")
  parser.add_argument("-o",
                      required=True,
                      help="path to folder to output image")
  args = parser.parse_args()

  # get all imzML files in input folder
  files = [file for file in os.listdir(args.i) if file.endswith(".imzML")]

  # list to store all mz lengths
  mzs_lengths = []

  # loop over all imzML files
  for file in files:
    # parse imzML file
    with ImzMLParser(os.path.join(args.i, file)) as p:
      # add mz lengths
      mzs_lengths.extend(p.mzLengths)

  # create dataframe for seaborn
  df = pd.DataFrame({"mzs_lengths": mzs_lengths})

  # create plot and save
  plt.figure(figsize=(15, 5))
  sns.histplot(data=df, x="mzs_lengths", bins=1000, kde=True, color="#001AFF")
  plt.xlabel("Mz Length")
  plt.ylabel("Frequency")
  plt.title("Mz Length Frequency Distribution")
  plt.savefig(os.path.join(args.o, "Mz Length Frequency Distribution.png"),
              transparent=True)
