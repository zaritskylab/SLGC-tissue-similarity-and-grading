"""Nanobiopsy classification analysis
This module should be imported and contains the following:
    
    * _map_record - Function to map a record to model input (spectra) and
            output (label).
    * _scale_spectra - Function to scale spectra.
    * _fixup_shape - Function to Fix the implicit inferring of the shapes of
            the output Tensors.
    * _create_ds - Function to create a dataset for model
    * _get_model - Function to generate classification model.
    * _spectras_info - Function to get all information except intensities
            (which needs a lot of memory) for each spectra from all images.
    * _train - Function to train models using leave one image and patient out.
    * _predict - Function to predict using leave one image and patient out.
    * classification_analysis - Function to apply a classification analysis.
"""

import os
import gc
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Union, Tuple
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import callbacks
from tensorflow.keras import metrics as k_metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pyimzml.ImzMLParser import ImzMLParser


def _map_record(file_name: tf.Tensor, idx: tf.Tensor,
                label: tf.Tensor) -> Tuple[np.ndarray, int]:
  """Function to map a record to model input (spectra) and output (label).

  Args:
      file_name (tf.Tensor): Record file name to get spectra.
      idx (tf.Tensor): Record index to get spectra.
      label (tf.Tensor): Record label.

  Returns:
      Tuple[np.ndarray, int]: Input (spectra) and output (label).
  
  """
  # Decoding from the EagerTensor object
  file_name, idx, label = (file_name.numpy(), idx.numpy(), label.numpy())

  # Decode bytes to str
  file_name = file_name.decode('utf-8')

  # Reading spectra from parser
  mzs, spectra = parsers[file_name].getspectrum(idx)

  # Return spectra and label
  return (spectra[((mzs >= 600) & (mzs <= 900))], label)


def _scale_spectra(
    x: tf.Tensor, y: tf.Tensor, min_spectra: np.ndarray, max_spectra: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Function to scale spectra.

  Args:
      x (tf.Tensor): Input (spectra)
      y (tf.Tensor): Output (label)
      min_spectra (np.ndarray): Min spectra to scale by.
      max_spectra (np.ndarray): Max spectra to scale by.

  Returns:
      Tuple[np.ndarray, np.ndarray]: Input (spectra) after scaling and 
          output (label) .
  
  """
  # Scale spectra
  x_scaled = (x - min_spectra) / (max_spectra - min_spectra)

  # Clip values between [0,1]
  x_scaled_clipped = np.clip(x_scaled, 0, 1)

  # Return scaled spectra after making sure its between 0 and 1 and label
  return x_scaled_clipped, y


def _fixup_shape(x: tf.Tensor, y: tf.Tensor):
  """ Function to Fix the implicit inferring of the shapes of the
      output Tensors.

  Args:
      x (tf.Tensor): Input (spectra)
      y (tf.Tensor): Output (label)

  Returns:
      Tuple[np.ndarray, np.ndarray]: Input (spectra) and output (label) with
        correct shape.
  
  """
  x.set_shape([24000])
  y.set_shape([])
  return x, y


def _create_ds(
    file_names: np.ndarray, indexes: np.ndarray, labels: np.ndarray,
    batch_size: int, shuffle: bool,
    min_max_spectra: Union[Tuple[np.ndarray, np.ndarray], None] = None
) -> tf.data.Dataset:
  """Function to create a dataset for model

  Args:
      file_names (np.ndarray): File names of the dataset.
      indexes (np.ndarray): Indexes of the dataset.
      labels (np.ndarray): Labels of the dataset.
      batch_size (int): Batch size.
      shuffle (bool): Flag to indicate if to shuffle or not.
      min_max_spectra (Tuple[np.ndarray,np.ndarray]): Min spectra and Max 
          spectra to apply scaling. Defaults to None (no scaling) 

  Returns:
      tf.data.Dataset: Dataset
  """
  # Create dataset
  ds = tf.data.Dataset.from_tensor_slices((file_names, indexes, labels))
  # Shuffle the data
  if shuffle:
    ds = ds.shuffle(len(file_names))
  # Map record to model input
  ds = ds.map(
      lambda i, j, k: tf.
      py_function(func=_map_record, inp=[i, j, k], Tout=[tf.float32, tf.int32])
  )
  # Scale record
  if min_max_spectra is not None:
    ds = ds.map(
        lambda i, j: tf.py_function(
            func=_scale_spectra, inp=[
                i, j, min_max_spectra[0], min_max_spectra[1]
            ], Tout=[tf.float32, tf.int32]
        )
    )
  # Fix the implicit inferring of the shapes of the output Tensors
  ds = ds.map(_fixup_shape)
  # Batch the spectra's
  ds = ds.batch(batch_size)
  # Prefetch batch's to make sure that a batch is ready to be served at all time
  ds = ds.prefetch(tf.data.AUTOTUNE)
  return ds


def _get_model() -> tf.keras.Model:
  """Function to generate classification model.

  Returns:
      tf.keras.Model: Classification model.
  
  """
  return tf.keras.Sequential(
      [
          layers.InputLayer(input_shape=(24000,)),
          layers.Dense(1024),
          layers.LeakyReLU(alpha=0.2),
          layers.BatchNormalization(),
          layers.Dropout(0.3),
          layers.Dense(1024),
          layers.LeakyReLU(alpha=0.2),
          layers.BatchNormalization(),
          layers.Dropout(0.3),
          layers.Dense(512),
          layers.LeakyReLU(alpha=0.2),
          layers.BatchNormalization(),
          layers.Dropout(0.3),
          layers.Dense(1, activation='sigmoid')
      ]
  )


def _spectras_info(processed_path: str, metadata: pd.DataFrame) -> pd.DataFrame:
  """Function to get all information except intensities
      (which needs a lot of memory) for each spectra from all images.

  Args:
      processed_path (str): Path to processed continuos imzML files.
      metadata (pd.DataFrame): Data frame of metadata.

  Returns:
      pd.DataFrame: All information except intensities for each spectra.
  """
  # Create lists to store each spectra's info
  spectras_info = []

  # Loop over each msi in the processed folder
  for _, row in metadata.iterrows():
    # Parse the msi file
    reader = parsers[row.sample_file_name]

    # Get segmentation image
    thresh_img = np.load(
        os.path.join(processed_path, row.sample_file_name, "segmentation.npy")
    )

    # Loop over each spectra
    for idx, (x, y, _) in enumerate(reader.coordinates):
      # Append spectra info
      spectras_info.append(
          [
              row.sample_file_name, row.sample_type, row.sample_number,
              row.histology, row.who_grade, row.label, x, y, idx,
              (True if thresh_img[y - 1, x - 1] else False)
          ]
      )

  # Convert to data frame
  spectras_info = pd.DataFrame(
      spectras_info, columns=[
          "file_name", "sample_type", "sample_number", "histology", "who_grade",
          "label", "x_coordinate", "y_coordinate", "idx", "is_tissue"
      ]
  )

  return spectras_info


def _train(
    metadata: pd.DataFrame, spectras_info: pd.DataFrame, output_path: str,
    biopsy_type: str, batch_size: int, learning_rate: float, epochs: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
  """Function to train models using leave one image and patient out. It will
      It will save for each left out the DNN and scaling models.

  Args:
      metadata (pd.DataFrame): Data frame of metadata.
      spectras_info (pd.DataFrame): All information except intensities for each
          spectra.
      output_path (str): Path to save classification models.
      biopsy_type (str): Type of biopsy. Can be 'section' or 'replica'
      batch_size (int): Batch size.
      learning_rate (float): Learning rate.
      epochs (int): Number of epochs.

  Returns:
      Tuple[pd.DataFrame, pd.DataFrame]: Validation and training metrics.
  """
  # Define dict's to store validation and training metrics
  train_metrics = {}
  validation_metrics = {}

  # Loop over each image
  for exclude_image, group in metadata.groupby("file_name"):
    # Clear graph
    K.clear_session()
    gc.collect()

    # Get all spectra's in the exclude_image to exclude them
    exclude_spectras = spectras_info["sample_number"].isin(
        group.sample_number.to_list()
    )

    # Create filter for training data - does not include the excluded image
    # and only include tissue spectra's
    train_filter = ((~exclude_spectras) & spectras_info.is_tissue)

    # Filter training data
    spectras_info_train = spectras_info.loc[train_filter]

    # Get x and y data for training
    X = spectras_info_train[["file_name", "idx"]].to_numpy()
    y = spectras_info_train["label"].to_numpy()

    # Split to train and validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # Create data generators
    train_generator = _create_ds(
        X_train[:, 0], X_train[:, 1].astype("int"), y_train, batch_size, True
    )

    # Create min max scaler object and train on training data
    scaler = MinMaxScaler()
    for batch in train_generator:
      batch = batch[0].numpy()
      scaler.partial_fit(batch)

    # Update train generator
    train_generator = _create_ds(
        X_train[:, 0], X_train[:, 1].astype("int"), y_train, batch_size, True,
        (scaler.data_min_, scaler.data_max_)
    )

    # Create validation generator
    validation_generator = _create_ds(
        X_val[:, 0], X_val[:, 1].astype("int"), y_val, batch_size, True,
        (scaler.data_min_, scaler.data_max_)
    )

    # Calculate class weights to make sure no imbalance data affect
    neg, pos = np.bincount(y_train.astype(int))
    weight_for_0 = (1 / neg) * ((neg + pos) / 2.0)
    weight_for_1 = (1 / pos) * ((neg + pos) / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}

    # Create Callback to save the best model
    checkpoint_filepath = os.path.join(
        output_path, "models", f"{biopsy_type}_excluded_{exclude_image}/"
    )
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, save_weights_only=False,
        monitor="val_auc", mode="max", save_best_only=True
    )

    # Create Callback for model early stopping
    model_es_callback = callbacks.EarlyStopping(
        monitor='val_loss', mode='min', verbose=1, patience=10
    )

    # Create classification model
    classification_model = _get_model()

    # Compile the classification model
    classification_model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(label_smoothing=0.1), metrics=[
            k_metrics.TruePositives(name='tp'),
            k_metrics.FalsePositives(name='fp'),
            k_metrics.TrueNegatives(name='tn'),
            k_metrics.FalseNegatives(name='fn'),
            k_metrics.BinaryAccuracy(name='accuracy'),
            k_metrics.Precision(name='precision'),
            k_metrics.Recall(name='recall'),
            k_metrics.AUC(name='auc'),
            k_metrics.AUC(name='prc', curve='PR')
        ]
    )

    # Train the classification model
    st = time.time()
    classification_model.fit(
        x=train_generator, validation_data=validation_generator, epochs=epochs,
        callbacks=[model_checkpoint_callback,
                   model_es_callback], class_weight=class_weight
    )
    elapsed_time = time.time() - st

    # Load the best saved
    classification_model = tf.keras.models.load_model(checkpoint_filepath)

    # Evaluate on train and validation
    train_metrics[exclude_image] = classification_model.evaluate(
        x=train_generator
    )
    train_metrics[exclude_image].append(elapsed_time)
    validation_metrics[exclude_image] = classification_model.evaluate(
        x=validation_generator
    )
    # Save scaler
    with open(
        os.path.join(
            output_path, "models",
            f"{biopsy_type}_excluded_{exclude_image}_scaler.pkl"
        ), 'wb'
    ) as f:
      pickle.dump(scaler, f)

    # Clean model for next iteration
    classification_model = None

    # Separate training
    print("#" * 30)

  # Create data frame of train metrics
  train_metrics_df = pd.DataFrame.from_dict(
      train_metrics, orient='index', columns=[
          "loss", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall",
          "auc", "prc", "train_time (seconds)"
      ]
  )

  # Create data frame of validation metrics
  validation_metrics_df = pd.DataFrame.from_dict(
      validation_metrics, orient='index', columns=[
          "loss", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall",
          "auc", "prc"
      ]
  )
  return train_metrics_df, validation_metrics_df


def _predict(
    metadata: pd.DataFrame, spectras_info: pd.DataFrame, output_path: str,
    biopsy_type: str, batch_size: int
) -> pd.DataFrame:
  """Function to predict using leave one image and patient out.

  Args:
      metadata (pd.DataFrame): Data frame of metadata.
      spectras_info (pd.DataFrame): All information except intensities for each
          spectra.
      output_path (str):  Path to saved classification models.
      biopsy_type (str): Type of biopsy. Can be 'section' or 'replica'.
      batch_size (int): Batch size.

  Returns:
      pd.DataFrame: spectras_info with added columns of prediction.
  """
  # Create copy of spectra info and add prediction column
  spectras_info = spectras_info.copy()
  spectras_info["prediction"] = -1

  # Loop over each sample number
  for _, row in metadata.iterrows():
    # Clear graph
    K.clear_session()
    gc.collect()

    # Create filter for test data
    test_filter = (spectras_info["file_name"] == row.sample_file_name)

    # Get x and y data for training
    X = spectras_info.loc[test_filter, ["file_name", "idx"]]
    y = spectras_info.loc[test_filter, "label"]

    # Open scaler
    with open(
        os.path.join(
            output_path, "models",
            f"{biopsy_type}_excluded_{row.file_name}_scaler.pkl"
        ), 'rb'
    ) as f:
      scaler = pickle.load(f)

    # Create test data generator
    test_generator = _create_ds(
        X.iloc[:, 0].to_numpy(), X.iloc[:, 1].to_numpy(), y.to_numpy(),
        batch_size, False, (scaler.data_min_, scaler.data_max_)
    )

    # Get saved model path
    model_path = os.path.join(
        output_path, "models", f"{biopsy_type}_excluded_{row.file_name}/"
    )

    # Load model
    classification_model = tf.keras.models.load_model(model_path)

    # Get predictions
    spectras_info.loc[test_filter, "prediction"] = classification_model.predict(
        x=test_generator
    )

    # Clean model for next iteration
    classification_model = None

  return spectras_info


def classification_analysis(
    processed_path: str, output_path: str, metadata: pd.DataFrame,
    batch_size: int = 256, learning_rate: float = 1e-3, epochs: int = 100
) -> None:
  """Function to apply a classification analysis.

  Args:
      processed_path (str): Path to processed continuos imzML files. 
      output_path (str): Path to save classification outputs.
      metadata (pd.DataFrame): Data frame of metadata.
      batch_size (int, optional): Batch size. Defaults to 1024.
      learning_rate (float, optional): Learning rate. Defaults to 1e-3.
      epochs (int, optional): Number of epochs. Defaults to 1.
  """
  # Define parser for each msi in order not to open every time we need to read
  # this has to be global because cant pass file handlers to tf.dataset.map
  # this is a workaround and not best practice
  global parsers
  parsers = {
      file_name:
      ImzMLParser(
          os.path.join(processed_path, file_name, "meaningful_signal.imzML")
      ) for file_name in metadata.sample_file_name.unique()
  }

  # Get all information except intensities (which needs a lot of memory) for
  # each spectra from all images
  spectras_info = _spectras_info(processed_path, metadata)

  # Split to section and replica
  s_spectras_info = spectras_info[spectras_info.sample_type == "section"]
  r_spectras_info = spectras_info[spectras_info.sample_type == "replica"]
  s_metadata_df = metadata[metadata.sample_type == "section"]
  r_metadata_df = metadata[metadata.sample_type == "replica"]

  # Train section models
  Path(os.path.join(output_path, "training")).mkdir(parents=True, exist_ok=True)
  train_metrics, validation_metrics = _train(
      s_metadata_df, s_spectras_info, output_path, "section", batch_size,
      learning_rate, epochs
  )
  train_metrics.to_csv(
      os.path.join(output_path, "training", "section_train_metrics.csv")
  )
  validation_metrics.to_csv(
      os.path.join(output_path, "training", "section_validation_metrics.csv")
  )

  # Train replica models
  train_metrics, validation_metrics = _train(
      r_metadata_df, r_spectras_info, output_path, "replica", batch_size,
      learning_rate, epochs
  )
  train_metrics.to_csv(
      os.path.join(output_path, "training", "replica_train_metrics.csv")
  )
  validation_metrics.to_csv(
      os.path.join(output_path, "training", "replica_validation_metrics.csv")
  )

  # Predict section using section models
  Path(os.path.join(output_path, "testing")).mkdir(parents=True, exist_ok=True)
  _predict(s_metadata_df, s_spectras_info, output_path, "section",
           batch_size).to_csv(
               os.path.join(
                   output_path, "testing",
                   "section_section_spectra_wise_predictions.csv"
               ), index=False
           )
  # Predict replica using replica models
  _predict(r_metadata_df, r_spectras_info, output_path, "replica",
           batch_size).to_csv(
               os.path.join(
                   output_path, "testing",
                   "replica_replica_spectra_wise_predictions.csv"
               ), index=False
           )

  # Create mapper from replica to section
  r_2_s = {
      'HG 12-11-r': "HG 11-11-12-s", 'HG 14-13-r': "HG 14-13-s", 'HG 16-15-r':
      "HG 16-15-s", 'HG 18-19-18-r': "HG 19-18-s", 'HG 29-25-23-21-20-r':
      "HG 29-25-23-21-20-s", 'HG 6-6-7-r': "HG 6-7-s", 'HG 8-5-4-3-2-r':
      "HG 8-12-5-4-3-2-s", 'HG 9-10-r': "HG 9-10-s", 'HG 1-r': "HG 1-s"
  }
  # Create mapper from section to replica
  s_2_r = {v: k for k, v in r_2_s.items()}
  # Map section to replica and replica to section
  s_metadata_df.loc[:,
                    "file_name"] = s_metadata_df.file_name.map(s_2_r.get).copy()
  r_metadata_df.loc[:,
                    "file_name"] = r_metadata_df.file_name.map(r_2_s.get).copy()
  # Predict section using replica models
  _predict(s_metadata_df, s_spectras_info, output_path, "replica",
           batch_size).to_csv(
               os.path.join(
                   output_path, "testing",
                   "replica_section_spectra_wise_predictions.csv"
               ), index=False
           )
  # Predict replica using section models
  _predict(r_metadata_df, r_spectras_info, output_path, "section",
           batch_size).to_csv(
               os.path.join(
                   output_path, "testing",
                   "section_replica_spectra_wise_predictions.csv"
               ), index=False
           )

  # Close parsers
  for reader in parsers.values():
    if reader.m:
      reader.m.close()
