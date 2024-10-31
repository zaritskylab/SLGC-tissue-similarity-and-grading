# Import the necessary libraries
import argparse
import json
import random
import warnings
import optuna
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Any, Union


def load_data(
    processed_files: List[Path], metadata_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
  """Function to load the data.

  Args:
    processed_files (List[Path]): List of processed files.
    metadata_df (pd.DataFrame): Metadata dataframe.

  Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
        np.ndarray]: A tuple containing the spectras, file names, sample 
        file names, sample numbers, sample types, and WHO grades.

  """
  # Define lists to store the data
  spectras = []
  file_names = []
  sample_file_names = []
  sample_numbers = []
  sample_types = []
  who_grades = []
  # Loop through the processed files
  for p in tqdm(
      processed_files, total=len(processed_files), desc="Loading data"
  ):
    # Get the spectras
    img = np.load(p / "mapped_tic_normalized.npy")
    seg = np.load(p / "segmentation.npy")
    spectras.append(img[seg])
    num_spectras = img[seg].shape[0]
    # Get the file name, sample file name, sample number, sample type, and
    # WHO grade
    metadata = metadata_df[metadata_df.sample_file_name == p.stem]
    file_name = metadata.file_name.values[0]
    sample_file_name = metadata.sample_file_name.values[0]
    sample_number = metadata.sample_number.values[0]
    sample_type = metadata.sample_type.values[0]
    who_grade = metadata.who_grade.values[0]
    # Append to the lists
    file_names.append([file_name] * num_spectras)
    sample_file_names.append([sample_file_name] * num_spectras)
    sample_numbers.append([sample_number] * num_spectras)
    sample_types.append([sample_type] * num_spectras)
    who_grades.append([who_grade] * num_spectras)
  # Convert lists to numpy arrays
  return (
      np.concatenate(spectras), np.concatenate(file_names),
      np.concatenate(sample_file_names), np.concatenate(sample_numbers),
      np.concatenate(sample_types), np.concatenate(who_grades)
  )


def convert_to_bulk(
    spectras: np.ndarray, file_names: np.ndarray, sample_file_names: np.ndarray,
    sample_numbers: np.ndarray, sample_types: np.ndarray,
    who_grades: np.ndarray, agg_func: str = 'mean'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray]:
  """Function to convert the data to bulk.

  Args:
    spectras (np.ndarray): Array of data spectras.
    file_names (np.ndarray): Array of file names.
    sample_file_names (np.ndarray): Array of sample file names.
    sample_numbers (np.ndarray): Array of sample numbers.
    sample_types (np.ndarray): Array of sample types.
    who_grades (np.ndarray): Array of WHO grades.
    agg_func (str): Aggregation function to use ('mean', 'max', 'median', 'min').

  Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
        np.ndarray]: A tuple containing the bulk spectras, file names, 
        sample_file_names, sample numbers, sample types, and WHO grades.

  """
  # Mapping of aggregation functions
  agg_functions = {
      'mean': np.mean, 'max': np.max, 'median': np.median, 'min': np.min
  }
  # Check if the specified aggregation function is valid
  if agg_func not in agg_functions:
    raise ValueError(
        f"Invalid aggregation function: {agg_func}. "
        f"Choose from {list(agg_functions.keys())}"
    )
  # Get the aggregation function based on agg_func
  aggregate = agg_functions[agg_func]
  # Create grouped indices by sample_file_name
  unique_sample_file_names = np.unique(sample_file_names)
  grouped_indices = {
      sample_file_name: np.where(sample_file_names == sample_file_name)[0]
      for sample_file_name in unique_sample_file_names
  }
  # Define lists to store the results
  bulk_spectras = []
  file_names_bulk = []
  sample_file_names_bulk = []
  sample_numbers_bulk = []
  sample_types_bulk = []
  who_grades_bulk = []
  # Iterate over each group to compute the bulk spectra and aggregate metadata
  for sample_file_name, indices in tqdm(
      grouped_indices.items(), desc="Converting to bulk"
  ):
    # Calculate the bulk of the spectra for this group
    bulk_spectrum = aggregate(spectras[indices], axis=0)
    # Extract metadata from the first index (as all values should be
    # identical within the group)
    file_name = file_names[indices[0]]
    sample_number = sample_numbers[indices[0]]
    sample_type = sample_types[indices[0]]
    who_grade = who_grades[indices[0]]
    # Append the results to the lists
    bulk_spectras.append(bulk_spectrum)
    file_names_bulk.append(file_name)
    sample_file_names_bulk.append(sample_file_name)
    sample_numbers_bulk.append(sample_number)
    sample_types_bulk.append(sample_type)
    who_grades_bulk.append(who_grade)
  # Return lists as numpy arrays
  return (
      np.array(bulk_spectras), np.array(file_names_bulk),
      np.array(sample_file_names_bulk), np.array(sample_numbers_bulk),
      np.array(sample_types_bulk), np.array(who_grades_bulk)
  )


def separate_data_by_sample_type(
    X: np.ndarray, y: np.ndarray, batch_ids: np.ndarray,
    patient_ids: np.ndarray, sample_types: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
  """Function to separate the data by sample type.

  Args:
    X (np.ndarray): The feature matrix.
    y (np.ndarray): The target vector.
    batch_ids (np.ndarray): The batch IDs.
    patient_ids (np.ndarray): The patient IDs.
    sample_types (np.ndarray): The sample types.

  Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
        np.ndarray, np.ndarray, np.ndarray]: A tuple containing the replica
        feature matrix, section feature matrix, replica target vector,
        section target vector, replica batch IDs, section batch IDs,
        replica patient IDs, and section patient IDs.

  """
  return (
      X[sample_types == 'replica'], X[sample_types == 'section'],
      y[sample_types == 'replica'], y[sample_types == 'section'
                                     ], batch_ids[sample_types == 'replica'],
      batch_ids[sample_types == 'section'
               ], patient_ids[sample_types == 'replica'
                             ], patient_ids[sample_types == 'section']
  )


def prepare_grouped_indices(batch_ids: np.ndarray) -> Dict[Any, np.ndarray]:
  """Function to prepare grouped indices for LOOCV.

  Args:
    batch_ids (np.ndarray): Array of batch IDs.

  Returns:
    Dict[Any, np.ndarray]: A dictionary containing the unique batch IDs as keys
        and the corresponding indices as values.

  """
  # Get unique values of batch IDs
  unique_vals = np.unique(batch_ids)
  # Create a dictionary to store the grouped indices
  grouped_indices = {val: np.where(batch_ids == val)[0] for val in unique_vals}
  return grouped_indices


def objective(
    trial, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
    batch_ids_train: np.ndarray, seed: int
) -> float:
  """
  Objective function for Optuna to optimize hyperparameters of a model using 
      cross-validation.

  Args:
    trial (optuna.trial.Trial): A trial object that suggests hyperparameters.
    X_train (np.ndarray): Training feature matrix of shape (n_samples, 
        n_features).
    y_train (np.ndarray): Training target vector of shape (n_samples,).
    batch_ids (np.ndarray): Array of group IDs used to group samples for 
        cross-validation.
    seed (int): Random seed for reproducibility.

  Returns:
    float: The mean AUC score across the cross-validation folds for the 
        suggested hyperparameters.

  """
  # Suggest hyperparameters using Optuna for Logistic Regression
  if model_type == 'logistic_regression':
    C = trial.suggest_float('C', 1e-6, 1e+6, log=True)
    max_iter = trial.suggest_int('max_iter', 100, 1000)
    tol = trial.suggest_float('tol', 1e-4, 1e-2, log=True)
    solver = trial.suggest_categorical('solver', ['liblinear', 'lbfgs'])
    model = LogisticRegression(
        C=C, max_iter=max_iter, tol=tol, solver=solver, class_weight='balanced',
        random_state=seed
    )
  # Suggest hyperparameters using Optuna for Decision Tree
  elif model_type == 'decision_tree':
    max_depth = trial.suggest_int('max_depth', 1, 32)
    max_features = trial.suggest_categorical(
        'max_features', ['sqrt', 'log2', None]
    )
    model = DecisionTreeClassifier(
        max_depth=max_depth, max_features=max_features, class_weight='balanced',
        random_state=seed
    )
  # Suggest hyperparameters using Optuna for Random Forest
  elif model_type == 'random_forest':
    class_weight = trial.suggest_categorical(
        'class_weight', ['balanced', 'balanced_subsample']
    )
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 7)
    max_features = trial.suggest_categorical(
        'max_features', ['sqrt', 'log2', None]
    )
    model = RandomForestClassifier(
        class_weight=class_weight, n_estimators=n_estimators,
        max_depth=max_depth, max_features=max_features, random_state=seed
    )
  # Suggest hyperparameters using Optuna for XGBoost
  elif model_type == 'xgboost':
    max_depth = trial.suggest_int('max_depth', 3, 7)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    subsample = trial.suggest_float('subsample', 0.6, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
    gamma = trial.suggest_float('gamma', 0, 5)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    model = XGBClassifier(
        max_depth=max_depth, learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, gamma=gamma,
        n_estimators=n_estimators, random_state=seed
    )
  # Suggest hyperparameters using Optuna for LightGBM
  else:
    max_depth = trial.suggest_int('max_depth', 3, 7)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
    num_leaves = trial.suggest_int('num_leaves', 30, 70)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    model = LGBMClassifier(
        num_leaves=num_leaves, learning_rate=learning_rate,
        n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced',
        random_state=seed, verbose=-1
    )
  # Define cross-validation
  # TODO: try with StratifiedKFold
  skf = StratifiedGroupKFold(n_splits=3, shuffle=False)
  # Define predictions array
  predictions = np.zeros(y_train.shape)
  # Perform cross-validation with the suggested hyperparameters
  for train_idx, val_idx in skf.split(X_train, y_train, batch_ids_train):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, _ = y_train[train_idx], y_train[val_idx]
    with warnings.catch_warnings():
      # Suppress specific warnings during hyperparameter tuning
      warnings.simplefilter("ignore", category=ConvergenceWarning)
      if model_type == 'xgboost':
        class_weights = compute_class_weight(
            'balanced', classes=np.unique(y_tr), y=y_tr
        )
        sample_weights = np.array([class_weights[int(y)] for y in y_tr])
        model.fit(X_tr, y_tr, sample_weight=sample_weights)
      else:
        model.fit(X_tr, y_tr)
    predictions[val_idx] = model.predict_proba(X_val)[:, 1]
  # Calculate AUC score and return
  return roc_auc_score(y_train, predictions)


def optimize_hyperparameters(
    X_train: np.ndarray, y_train: np.ndarray, batch_ids_train: np.ndarray,
    model_type: str, seed: int, n_trials: int = 50, n_jobs: int = -1
) -> Dict[str, Any]:
  """Function to optimize hyperparameters using Optuna.

  Args:
    X_train (np.ndarray): Training feature matrix.
    y_train (np.ndarray): Training target vector.
    batch_ids_train (np.ndarray): Array of group IDs used to group samples for
        cross-validation.
    model_type (str): The type of model to optimize hyperparameters for.
    seed (int): Random seed for reproducibility.
    n_trials (int, optional): Number of trials for hyperparameter optimization.
    n_jobs (int, optional): Number of parallel jobs to run.
  
  Returns:
    Dict[str, Any]: A dictionary containing the best hyperparameters found

  """
  study = optuna.create_study(direction='maximize')
  study.optimize(
      lambda trial:
      objective(trial, model_type, X_train, y_train, batch_ids_train, seed),
      n_trials=n_trials, n_jobs=n_jobs
  )
  return study.best_params


def create_best_model(
    model_type: str, best_params: Dict[str, Any], seed: int
) -> Union[LogisticRegression, DecisionTreeClassifier, XGBClassifier,
           RandomForestClassifier, LGBMClassifier]:
  """Function to create the best model using the best hyperparameters.

  Args:
    model_type (str): he type of model to optimize hyperparameters for.
    best_params (Dict[str, Any]): Dictionary of best hyperparameters found.
    seed (int): Random seed for reproducibility.

  Returns:
    Union[LogisticRegression, DecisionTreeClassifier, XGBClassifier, 
        RandomForestClassifier, LGBMClassifier]: The best model.

  """
  # Create the best model for Logistic Regression
  if model_type == 'logistic_regression':
    best_model = LogisticRegression(**best_params, random_state=seed)
  # Create the best model for Decision Tree
  elif model_type == 'decision_tree':
    best_model = DecisionTreeClassifier(**best_params, random_state=seed)
  # Create the best model for XGBoost
  elif model_type == 'xgboost':
    best_model = XGBClassifier(**best_params, random_state=seed)
  # Create the best model for Random Forest
  elif model_type == 'random_forest':
    best_model = RandomForestClassifier(**best_params, random_state=seed)
  # Create the best model for LightGBM
  else:
    best_model = LGBMClassifier(**best_params, random_state=seed, verbose=-1)
  return best_model


def fit_and_calibrate_model(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    model_type: str, best_params: Dict[str, Any], seed: int
) -> np.ndarray:
  """Function to fit and calibrate a model using the best hyperparameters.

  Args:
    X_train (np.ndarray): Training feature matrix.
    y_train (np.ndarray): Training target vector.
    X_test (np.ndarray): Test feature matrix.
    model_type (str): The type of model to optimize hyperparameters for.
    best_params (Dict[str, Any]): Dictionary of best hyperparameters found.
    seed (in): Random seed for reproducibility.

  Returns:
    np.ndarray: Predictions for the test set.

  """
  # Create and fit the model using the best parameters
  best_model = create_best_model(model_type, best_params, seed)
  if model_type == 'xgboost':
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    sample_weights = np.array([class_weights[int(y)] for y in y_train])
    best_model.fit(X_train, y_train, sample_weight=sample_weights)
  else:
    best_model.fit(X_train, y_train)
  # Calibrate the model
  calibrated_classifier = CalibratedClassifierCV(
      best_model, method='sigmoid', cv='prefit'
  )
  calibrated_classifier.fit(X_train, y_train)
  # Make predictions
  return calibrated_classifier.predict_proba(X_test)[:, 1]


def train_and_predict_for_group(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    X_test_other: np.ndarray, batch_ids_train: np.ndarray, model_type: str,
    seed: int, best_params: Dict[str, Any] = None, n_trials: int = 50,
    n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
  """Function to train and predict for a group of samples.

  Args:
    X_train (np.ndarray): Training feature matrix.
    y_train (np.ndarray): Training target vector.
    X_test (np.ndarray): Test feature matrix.
    X_test_other (np.ndarray): Test feature matrix for the other sample type.
    batch_ids_train (np.ndarray): Array of group IDs used to group samples for
        cross-validation.
    model_type (str): The type of model to optimize hyperparameters for.
    seed (int): Random seed for reproducibility.
    best_params (Dict[str, Any], optional): Dictionary of best hyperparameters
        found. Defaults to None.
    n_trials (int, optional): Number of trials for hyperparameter optimization.
    n_jobs (int, optional): Number of parallel jobs to run.

  Returns:
    Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: A tuple containing the
        predictions for the replica samples, predictions for the section
        samples, and the best hyperparameters found.

  """
  # Get the best hyperparameters if not provided
  if best_params is None:
    best_params = optimize_hyperparameters(
        X_train, y_train, batch_ids_train, model_type, seed, n_trials, n_jobs
    )
  # Combine X_test and X_test_other for prediction
  X_test_combined = np.concatenate([X_test, X_test_other])
  # Predict for combined data
  preds_combined = fit_and_calibrate_model(
      X_train, y_train, X_test_combined, model_type, best_params, seed
  )
  # Separate the predictions for replica and section and return them
  return preds_combined[:len(X_test)], preds_combined[len(X_test):], best_params


def perform_loocv(
    X: np.ndarray, y: np.ndarray, batch_ids: np.ndarray,
    patient_ids: np.ndarray, other_X: np.ndarray, other_patient_ids: np.ndarray,
    model_type: str, seed: int, best_params_list: List[Dict[str, Any]] = None,
    n_trials: int = 50, n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
  """ Function to perform leave-one-out cross-validation.

  Args:
    X (np.ndarray): Training feature matrix.
    y (np.ndarray): Training target vector.
    batch_ids (np.ndarray): Array of group IDs used to group samples for
    patient_ids (np.ndarray): Array of patient IDs.
    other_X (np.ndarray): Test feature matrix for the other sample type.
    other_patient_ids (np.ndarray): Array of patient IDs for the other sample
    model_type (str): The type of model to optimize hyperparameters for.
    seed (int): Random seed for reproducibility.
    best_params_list (List[Dict[str, Any]], optional): List of best 
        hyperparameters found. Defaults to None.
    n_trials (int, optional): Number of trials for hyperparameter optimization.
    n_jobs (int, optional): Number of parallel jobs to run.

  Returns:
    Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]: A tuple containing the
        predicted probabilities for the replica samples, predicted probabilities
        for the section samples, and the best hyperparameters used.

  """
  # Prepare grouped indices for LOOCV
  grouped_indices = prepare_grouped_indices(batch_ids)
  # Arrays to store predicted probabilities and best parameters used
  predicted_probabilities = np.zeros(X.shape[0])
  predicted_probabilities_cross = np.zeros(other_X.shape[0])
  best_params_used = []
  # Loop over each group for training and testing
  for idx, (_, test_idx) in tqdm(
      enumerate(grouped_indices.items()), total=len(grouped_indices),
      desc="LOOCV"
  ):
    # Define train indices by excluding patients in the test set
    train_idx = ~np.isin(patient_ids, patient_ids[test_idx])
    # Identify cross-test indices in the other sample type that correspond to
    # the same patients
    cross_test_idx = np.isin(other_patient_ids, patient_ids[test_idx])
    # Get the best parameters if provided, otherwise train and optimize
    best_params = best_params_list[idx] if best_params_list else None
    # Perform training and prediction
    preds, preds_cross, best_params = train_and_predict_for_group(
        X[train_idx], y[train_idx], X[test_idx], other_X[cross_test_idx],
        batch_ids[train_idx], model_type, seed, best_params, n_trials, n_jobs
    )
    # Store the predictions
    predicted_probabilities[test_idx] = preds
    predicted_probabilities_cross[cross_test_idx] = preds_cross
    best_params_used.append(best_params)
  # Return the predicted probabilities and the best parameters used
  return (
      predicted_probabilities, predicted_probabilities_cross, best_params_used
  )


def single_seed_classification(
    X: np.ndarray, y: np.ndarray, batch_ids: np.ndarray,
    patient_ids: np.ndarray, sample_types: np.ndarray, model_type: str,
    seed: int, best_params_r: Dict[str, Any] = None,
    best_params_s: Dict[str, Any] = None, n_trials: int = 50, n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[
    str, Any]]:
  """Function to perform classification using a single seed.

  Args:
    X (np.ndarray): Training feature matrix.
    y (np.ndarray): Training target vector.
    batch_ids (np.ndarray): Array of group IDs used to group samples for
    patient_ids (np.ndarray): Array of patient IDs.
    sample_types (np.ndarray): Array of sample types.
    model_type (str): The type of model to optimize hyperparameters for.
    seed (int): Random seed for reproducibility.
    best_params_r (Dict[str, Any]): Best hyperparameters for replica samples. 
        Defaults to None.
    best_params_s (Dict[str, Any]): Best hyperparameters for section samples. 
        Defaults to None.
    n_trials (int, optional): Number of trials for hyperparameter optimization.
    n_jobs (int, optional): Number of parallel jobs to run.

  Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any],
          Dict[str, Any]]: A tuple containing the predicted probabilities for
          the replica samples, predicted probabilities for the section samples,
          predicted probabilities for replica samples using section models,
          predicted probabilities for section samples using replica models,
          best hyperparameters for replica samples, and best hyperparameters for
          section samples.

  """
  # Separate the data based on the sample type
  (X_r, X_s, y_r, y_s, batch_ids_r, batch_ids_s, patient_ids_r, patient_ids_s
  ) = separate_data_by_sample_type(X, y, batch_ids, patient_ids, sample_types)
  # Perform LOOCV for replicas and sections
  (predicted_probabilities_r, predicted_probabilities_rs,
   best_params_r) = perform_loocv(
       X_r, y_r, batch_ids_r, patient_ids_r, X_s, patient_ids_s, model_type,
       seed, best_params_r, n_trials, n_jobs
   )
  (predicted_probabilities_s, predicted_probabilities_sr,
   best_params_s) = perform_loocv(
       X_s, y_s, batch_ids_s, patient_ids_s, X_r, patient_ids_r, model_type,
       seed, best_params_s, n_trials, n_jobs
   )
  # Return the predicted probabilities
  return (
      predicted_probabilities_r, predicted_probabilities_s,
      predicted_probabilities_rs, predicted_probabilities_sr, best_params_r,
      best_params_s
  )


def single_seed_bulk_and_non_bulk_classification(
    seed: int, spectras: np.ndarray, file_names: np.ndarray,
    sample_numbers: np.ndarray, sample_types: np.ndarray,
    who_grades: np.ndarray, model_type: str, n_trials: int = 50,
    n_jobs: int = -1, output_dir: str = "output"
) -> None:
  """Function to run bulk and non-bulk classification with a single seed.

  Args:
    seed (int): Seed for reproducibility.
    spectras (np.ndarray): Spectras for non-bulk samples.
    file_names (np.ndarray): File names for non-bulk samples.
    sample_numbers (np.ndarray): Sample numbers for non-bulk samples.
    sample_types (np.ndarray): Sample types for non-bulk samples.
    who_grades (np.ndarray): WHO grades for non-bulk samples.
    model_type (str): The type of model to optimize hyperparameters for.
    n_trials (int, optional): Number of trials for hyperparameter optimization.
    n_jobs (int, optional): Number of parallel jobs to run.
    output_dir (str, optional): Output directory to save the results.

  """
  # Set the seed for reproducibility
  np.random.seed(seed)
  random.seed(seed)
  # Create a folder for the seed
  seed_dir = output_dir / f"seed_{seed}"
  seed_dir.mkdir(parents=True, exist_ok=True)
  # Run non bulk classification
  (
      predicted_probabilities_r, predicted_probabilities_s,
      predicted_probabilities_rs, predicted_probabilities_sr, best_params_r,
      best_params_s
  ) = single_seed_classification(
      spectras.copy(), (who_grades > 2).astype(int), file_names.copy(),
      sample_numbers.copy(), sample_types.copy(), model_type, seed, None, None,
      n_trials, n_jobs
  )
  # Save non-bulk results
  np.save(seed_dir / "predicted_probabilities_r.npy", predicted_probabilities_r)
  np.save(seed_dir / "predicted_probabilities_s.npy", predicted_probabilities_s)
  np.save(
      seed_dir / "predicted_probabilities_rs.npy", predicted_probabilities_rs
  )
  np.save(
      seed_dir / "predicted_probabilities_sr.npy", predicted_probabilities_sr
  )
  with open(seed_dir / "best_params_r.json", 'w') as f:
    json.dump(best_params_r, f)
  with open(seed_dir / "best_params_s.json", 'w') as f:
    json.dump(best_params_s, f)


def multiple_seeds_classification_with_parallel(
    primary_seed: int, iterations: int, model_type: str,
    processed_files: List[Path], metadata_df: pd.DataFrame, output_dir: Path,
    n_trials: int = 50, n_jobs: int = -1
) -> List[int]:
  """Function to run bulk and non-bulk classification with multiple seeds in 
    parallel.

  Args:
    primary_seed (int): Seed for reproducibility.
    iterations (int): Number of iterations to run.
    model_type (str): Type of model to use for classification.
    processed_files (List[Path]): List of processed files.
    metadata_df (pd.DataFrame): Metadata dataframe.
    output_dir (Path): Output directory to save the results.
    n_trials (int, optional): Number of trials for hyperparameter optimization.
    n_jobs (int, optional): Number of parallel jobs to run.

  Returns:
    List[int]: List of seeds used for each classification.
  
  """
  # Set the primary seed for reproducibility
  np.random.seed(primary_seed)
  random.seed(primary_seed)
  # Load the data
  (
      spectras, file_names, sample_file_names, sample_numbers, sample_types,
      who_grades
  ) = load_data(processed_files, metadata_df)
  # Generate multiple seeds for evaluation
  evaluation_seeds = [primary_seed] + [
      int(i) for i in
      np.random.choice(range(10000), size=iterations - 1, replace=False)
  ]
  # Create the output directory if it does not exist
  output_dir_path = Path(output_dir)
  output_dir_path.mkdir(parents=True, exist_ok=True)
  # Use joblib to run the function in parallel for each seed
  results = [
      r for r in tqdm(
          Parallel(return_as="generator", n_jobs=n_jobs)(
              delayed(single_seed_bulk_and_non_bulk_classification)(
                  seed=seed, spectras=spectras, file_names=file_names,
                  sample_numbers=sample_numbers, sample_types=sample_types,
                  who_grades=who_grades, model_type=model_type,
                  n_trials=n_trials, n_jobs=n_jobs, output_dir=output_dir_path
              ) for seed in evaluation_seeds
          ), total=len(evaluation_seeds),
          desc="Running classification for multiple seeds"
      )
  ]

  # Return the list of seeds used for reference
  return evaluation_seeds


# Define current folder using this file
CWD = Path(".")
# Define folder that contains dataset
DHG_PATH = CWD / ".." / "data" / "DHG"
# Define folder that contains raw data
RAW_DATA = DHG_PATH / "raw_txt"
# Define folder to save processed data
PROCESSED_DATA = DHG_PATH / "processed_txt"
# Define file that contains metadata
METADATA_PATH = DHG_PATH / "txt_metadata.csv"
# Define path to save plots and results
FIGURES_PATH = CWD / "new_correlation_classification"
FIGURES_PATH.mkdir(exist_ok=True, parents=True)
# Define mass range start value
MZ_START = 600
# Define mass range end value
MZ_END = 900
# Define mass resolution of the data
MASS_RESOLUTION = 0.02
# Define representative peaks
REPRESENTATIVE_PEAKS = [794.5, 834.5, 886.6]
# Define the primary seed for reproducibility
PRIMARY_SEED = 42
# Read metadata csv
metadata_df = pd.read_csv(METADATA_PATH)

if __name__ == "__main__":
  # Parse command-line arguments
  parser = argparse.ArgumentParser(description="Classification pipeline")
  parser.add_argument(
      '--n_iterations', type=int, default=10, help="Number of iterations to run"
  )
  parser.add_argument(
      '--n_permutations', type=int, default=1000,
      help="Number of permutations for the analysis"
  )
  parser.add_argument(
      '--model_type', type=str, choices=[
          'logistic_regression', 'decision_tree', 'random_forest', 'lightgbm',
          'xgboost'
      ], default='lightgbm', help=(
          "Type of model to use (e.g., 'logistic', 'decision_tree',"
          "'random_forest', 'lightgbm' or 'xgboost')"
      )
  )
  args = parser.parse_args()

  # Get the processed files
  processed_files = list(Path(PROCESSED_DATA).iterdir())

  # Define the output path
  output_path = FIGURES_PATH / "classification" / args.model_type / "no_bulk"
  output_path.mkdir(parents=True, exist_ok=True)
  # Run the classification with multiple seeds in parallel
  evaluation_seeds = multiple_seeds_classification_with_parallel(
      PRIMARY_SEED, args.n_iterations, args.model_type, processed_files,
      metadata_df, output_path, n_trials=50, n_jobs=-1
  )
