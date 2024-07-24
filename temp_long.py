import sys

sys.path.append(".")

import os
import pickle
import numpy as np
import pandas as pd
from utils import read_msi
from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.ImzMLParser import ImzMLParser
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
from processing import MeanSegmentation, ZScoreCorrection
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap

# Define current folder using this file
CWD = "."
# Define folder that contains the revision chip type dataset
BASE_PATH = Path(os.path.join(CWD, "..", "data", "LONGITUDINAL"))
# Define folder that contains raw data
RAW_DATA = BASE_PATH.joinpath("raw")
# Define folder to save aligned data
ALIGNED_DATA = BASE_PATH.joinpath("aligned")
# Define folder to save processed data
PROCESSED_DATA = BASE_PATH.joinpath("processed")


# Run HCA analysis
def perform_hierarchical_clustering(pixels, n_clusters):
  # Perform hierarchical clustering
  clustering = AgglomerativeClustering(
      n_clusters=n_clusters, linkage='ward', compute_distances=True
  )
  clustering.fit(pixels)
  return clustering


def plot_dendrogram(model, path, name, **kwargs):
  # Create linkage matrix and plot the dendrogram
  counts = np.zeros(model.children_.shape[0])
  n_samples = len(model.labels_)
  for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
      if child_idx < n_samples:
        current_count += 1
      else:
        current_count += counts[child_idx - n_samples]
    counts[i] = current_count

  linkage_matrix = np.column_stack([model.children_, model.distances_,
                                    counts]).astype(float)

  fig, ax = plt.subplots(figsize=(15, 10))
  dendrogram(linkage_matrix, **kwargs)
  for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize('14')
    label.set_color('0.2')
  for spine in ['bottom', 'left']:
    ax.spines[spine].set_linewidth(2.5)
    ax.spines[spine].set_color('0.2')
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.tight_layout()
  plt.savefig(
      path / f"{name}_dendrogram.png", bbox_inches='tight', dpi=1200,
      transparent=True
  )
  plt.show()


def plot_clusters(labels, mask, original_shape, path, name):
  #
  tab10_cmap = plt.get_cmap('tab10')
  selected_colors = [tab10_cmap(i) for i in range(n_clusters)]
  colors = np.vstack(([1, 1, 1, 1], selected_colors))
  custom_cmap = ListedColormap(colors)

  # Reshape labels to original image shape
  label_image = np.zeros(original_shape)
  label_image[mask] = labels
  #flipped_label_image = np.fliplr(label_image)
  seg_c_map = plt.imshow(label_image, cmap=custom_cmap, vmin=0, vmax=n_clusters)
  cbar = plt.colorbar(
      seg_c_map, ticks=np.arange(1, n_clusters + 1),
      boundaries=np.arange(0.5, n_clusters + 1.5)
  )
  cbar.set_ticklabels(range(1, n_clusters + 1))
  cbar.set_label(
      'Cluster', labelpad=15, fontweight='bold', fontsize=14, color='0.2'
  )
  cbar.outline.set_edgecolor('0.2')
  cbar.ax.tick_params(labelsize=14, width=2.5, color='0.2')
  for l in cbar.ax.get_yticklabels():
    l.set_fontweight('bold')
    l.set_color('0.2')
  plt.axis('off')
  plt.tight_layout()
  plt.savefig(
      path / f"{name}_clusters.png", bbox_inches='tight', dpi=1200,
      transparent=True
  )
  plt.show()


def find_top_mz_values_per_cluster(
    img_filtered, labels, mzs_filtered, n_clusters, top_n=30
):
  # Initialize an array to store the mean intensities of m/z values for each cluster
  cluster_means = [
      np.mean(img_filtered[labels == i], axis=0)
      for i in range(1, n_clusters + 1)
  ]

  # Find the top 30 m/z values in each cluster and collect their indices
  top_indices = [
      np.argsort(cluster_mean)[-top_n:] for cluster_mean in cluster_means
  ]

  # Retrieve the m/z values corresponding to these indices
  top_mz_values = [mzs_filtered[indices] for indices in top_indices]

  # Combine and find unique m/z values across all clusters
  unique_mz_values = np.unique(np.concatenate(top_mz_values))

  # Create a dictionary to map m/z values to indices for easy lookup
  mz_to_index = {mz: idx for idx, mz in enumerate(unique_mz_values)}

  # Create a heatmap data matrix
  heatmap_data = np.zeros((len(unique_mz_values), n_clusters))

  # Fill the heatmap data matrix
  for cluster_index, indices in enumerate(top_indices):
    for idx in indices:
      mz_value = mzs_filtered[idx]
      heatmap_data[mz_to_index[mz_value],
                   cluster_index] = cluster_means[cluster_index][idx]

  return heatmap_data, unique_mz_values


def plot_heatmap(heatmap_data, unique_mz_values, n_clusters, path, name):
  fig, ax = plt.subplots(figsize=(5, 10))
  im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')

  # Create colorbar
  cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046)
  cbar.outline.set_edgecolor('0.2')
  cbar.set_ticks([])

  # We want to show all ticks...
  ax.set_xticks(np.arange(n_clusters))
  ax.set_yticks(np.arange(len(unique_mz_values)))

  # ... and label them with the respective list entries
  ax.set_xticklabels(range(1, n_clusters + 1))
  ax.set_yticklabels(np.round(unique_mz_values, decimals=2))

  # Turn spines off and create white grid.
  ax.spines[:].set_visible(False)
  ax.set_xticks(np.arange(heatmap_data.shape[1] + 1) - .5, minor=True)
  ax.set_yticks(np.arange(heatmap_data.shape[0] + 1) - .5, minor=True)
  ax.tick_params(which="minor", size=0)

  plt.xlabel(
      'Cluster', labelpad=15, fontweight='bold', fontsize=14, color='0.2'
  )
  plt.ylabel('m/z', labelpad=15, fontweight='bold', fontsize=14, color='0.2')
  plt.tight_layout()
  plt.savefig(
      path / f"{name}_heatmap.png", bbox_inches='tight', dpi=1200,
      transparent=True
  )
  plt.show()


n_clusters = 3
save_path_1 = Path(".") / "longitudinal" / "hca_gaussian_smoothing"
save_path_1.mkdir(parents=True, exist_ok=True)
save_path_2 = Path(".") / "longitudinal" / "hca_median_smoothing"
save_path_2.mkdir(parents=True, exist_ok=True)
save_path_3 = Path(".") / "longitudinal" / "hca"
save_path_3.mkdir(parents=True, exist_ok=True)

for folder in PROCESSED_DATA.iterdir():
  if 'tmz' in folder.name and "tumor" in folder.name and 'brain_6' in folder.name:
    print(f"Working {folder.name}")

    msi_path = folder / 'meaningful_signal.imzML'
    msi_seg = np.load(folder / "segmentation_new.npy")

    with ImzMLParser(msi_path) as p:
      mzs, img = read_msi(p)
      mzs_filter = (mzs >= 600) & (mzs <= 900)
      img_filter = img[:, :, mzs_filter]

      from scipy import ndimage
      smoothed_img = ndimage.gaussian_filter(img_filter, sigma=0.5)

      # Cluster the pixels
      cluster_model = perform_hierarchical_clustering(
          smoothed_img[msi_seg], n_clusters
      )

      with open(save_path_1 / f"{folder.name}_hca.pkl", 'wb') as f:
        pickle.dump(cluster_model, f)

      # Plot dendrogram
      plot_dendrogram(
          cluster_model, save_path_1, folder.name, truncate_mode='level', p=3
      )

      # Plot clusters
      plot_clusters(
          cluster_model.labels_ + 1, msi_seg, img_filter.shape[:-1],
          save_path_1, folder.name
      )

      heatmap_data, unique_mz_values = find_top_mz_values_per_cluster(
          img_filter[msi_seg], cluster_model.labels_ + 1, mzs[mzs_filter],
          n_clusters
      )

      plot_heatmap(
          heatmap_data, unique_mz_values, n_clusters, save_path_1, folder.name
      )

      smoothed_img = ndimage.median_filter(img_filter, 3)

      # Cluster the pixels
      cluster_model = perform_hierarchical_clustering(
          smoothed_img[msi_seg], n_clusters
      )

      with open(save_path_2 / f"{folder.name}_hca.pkl", 'wb') as f:
        pickle.dump(cluster_model, f)

      # Plot dendrogram
      plot_dendrogram(
          cluster_model, save_path_2, folder.name, truncate_mode='level', p=3
      )

      # Plot clusters
      plot_clusters(
          cluster_model.labels_ + 1, msi_seg, img_filter.shape[:-1],
          save_path_2, folder.name
      )

      heatmap_data, unique_mz_values = find_top_mz_values_per_cluster(
          img_filter[msi_seg], cluster_model.labels_ + 1, mzs[mzs_filter],
          n_clusters
      )

      plot_heatmap(
          heatmap_data, unique_mz_values, n_clusters, save_path_2, folder.name
      )

      # Cluster the pixels
      cluster_model = perform_hierarchical_clustering(
          img_filter[msi_seg], n_clusters
      )

      with open(save_path_3 / f"{folder.name}_hca.pkl", 'wb') as f:
        pickle.dump(cluster_model, f)

      # Plot dendrogram
      plot_dendrogram(
          cluster_model, save_path_3, folder.name, truncate_mode='level', p=3
      )

      # Plot clusters
      plot_clusters(
          cluster_model.labels_ + 1, msi_seg, img_filter.shape[:-1],
          save_path_3, folder.name
      )

      heatmap_data, unique_mz_values = find_top_mz_values_per_cluster(
          img_filter[msi_seg], cluster_model.labels_ + 1, mzs[mzs_filter],
          n_clusters
      )

      plot_heatmap(
          heatmap_data, unique_mz_values, n_clusters, save_path_3, folder.name
      )
