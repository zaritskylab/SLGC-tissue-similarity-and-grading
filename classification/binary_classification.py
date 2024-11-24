"""Binary Classification module.
This module can be ran as a script to perform binary classification on the DHG
dataset or imported as a module to use the functions. Contains the following:
    * main - Function to run binary classification on the DHG dataset.

"""
import os
import argparse
import json
import random
import warnings
import optuna
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import figure_customizer as fc
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
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
  for p in processed_files:
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
        C=C, max_iter=max_iter, tol=tol, solver=solver, random_state=seed
    )
  # Suggest hyperparameters using Optuna for Random Forest
  elif model_type == 'random_forest':
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 7)
    max_features = trial.suggest_categorical(
        'max_features', ['sqrt', 'log2', None]
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        max_features=max_features, random_state=seed
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
    num_leaves = trial.suggest_int('num_leaves', 15, 50)
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    feature_fraction = trial.suggest_float('feature_fraction', 0.6, 1.0)
    model = LGBMClassifier(
        num_leaves=num_leaves, learning_rate=learning_rate,
        feature_fraction=feature_fraction, n_estimators=n_estimators,
        max_depth=max_depth, random_state=seed, verbose=-1
    )
  # Define cross-validation
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
) -> Union[LogisticRegression, XGBClassifier, RandomForestClassifier,
           LGBMClassifier]:
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


def fit_and_predict_model(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    model_type: str, best_params: Dict[str, Any], seed: int
) -> np.ndarray:
  """Function to fit and predict a model using the best hyperparameters.

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
  best_model.fit(X_train, y_train)
  # Make predictions
  return best_model.predict_proba(X_test)[:, 1]


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
  preds_combined = fit_and_predict_model(
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


def single_seed_loocv(
    X: np.ndarray, y: np.ndarray, batch_ids: np.ndarray,
    patient_ids: np.ndarray, sample_types: np.ndarray, model_type: str,
    seed: int, best_params_r: Dict[str, Any] = None,
    best_params_s: Dict[str, Any] = None, n_trials: int = 50, n_jobs: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], Dict[
    str, Any]]:
  """Function to perform loocv classification using a single seed.

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


def single_seed_classification(
    seed: int, spectras: np.ndarray, file_names: np.ndarray,
    sample_numbers: np.ndarray, sample_types: np.ndarray,
    who_grades: np.ndarray, model_type: str,
    best_params_r: Dict[str, Any] = None, best_params_s: Dict[str, Any] = None,
    n_trials: int = 50, n_jobs: int = -1, output_dir: str = "output"
) -> None:
  """Function to run classification with a single seed.

  Args:
    seed (int): Seed for reproducibility.
    spectras (np.ndarray): Spectras. 
    file_names (np.ndarray): File names.
    sample_numbers (np.ndarray): Sample numbers.
    sample_types (np.ndarray): Sample types.
    who_grades (np.ndarray): WHO grades.
    model_type (str): The type of model to optimize hyperparameters for.
    best_params_r (Dict[str, Any], optional): Best hyperparameters for replica
        samples. Defaults to None.
    best_params_s (Dict[str, Any], optional): Best hyperparameters for section
        samples. Defaults to None.
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
  # Run classification
  (
      predicted_probabilities_r, predicted_probabilities_s,
      predicted_probabilities_rs, predicted_probabilities_sr, best_params_r,
      best_params_s
  ) = single_seed_loocv(
      spectras.copy(), (who_grades > 2).astype(int), file_names.copy(),
      sample_numbers.copy(), sample_types.copy(), model_type, seed,
      best_params_r, best_params_s, n_trials, n_jobs
  )
  # Save results
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
  """Function to run classification with multiple seeds in parallel.

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
              delayed(single_seed_classification)(
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


def get_best_params_from_best_seed(
    model_output_dir: Path, who_grades_r: np.ndarray, who_grades_s: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """ Function to get the best parameters from the best seed.

  Args:
    model_output_dir (Path): Path to the output directory containing model 
        saved results.
    who_grades_r (np.ndarray): WHO grades for non-bulk data (replica).
    who_grades_s (np.ndarray): WHO grades for non-bulk data (section).
  
  Returns:
    Tuple[Dict[str, Any], Dict[str, Any]]: Best parameters for replica and 
        section data.

  """
  # Initialize lists to store true labels and predictions
  y_true_r = (who_grades_r > 2).astype(int)
  y_true_s = (who_grades_s > 2).astype(int)
  # Define variables to store the best AUC scores
  best_auc_r, best_auc_s, best_auc_rs, best_auc_sr = (
      -np.inf, -np.inf, -np.inf, -np.inf
  )
  # Initialize best parameters
  best_params_r, best_params_s = None, None
  #
  seed_r, seed_s = None, None
  # Loop through each seed and load the saved predictions
  for seed_dir in model_output_dir.glob("seed_*"):
    if len(list(seed_dir.glob("*.npy"))) == 0:
      continue
    # Load predicted probabilities
    pred_r = np.load(seed_dir / "predicted_probabilities_r.npy")
    pred_s = np.load(seed_dir / "predicted_probabilities_s.npy")
    # Calculate ROC curves for each category
    auc_r = roc_auc_score(y_true_r, pred_r)
    auc_s = roc_auc_score(y_true_s, pred_s)
    # Check if the current model is better than the previous best model
    if auc_r > best_auc_r:
      best_auc_r = auc_r
      best_params_r = json.load(open(seed_dir / "best_params_r.json"))
      seed_r = int(seed_dir.stem.split("_")[-1])
    # Check if the current model is better than the previous best model
    if auc_s > best_auc_s:
      best_auc_s = auc_s
      best_params_s = json.load(open(seed_dir / "best_params_s.json"))
      seed_s = int(seed_dir.stem.split("_")[-1])
  return best_params_r, best_params_s, seed_r, seed_s


def single_seed_permutation(
    seed: int, spectras: np.ndarray, file_names: np.ndarray,
    sample_numbers: np.ndarray, sample_types: np.ndarray,
    who_grades: np.ndarray, model_type: str, best_params_r: Dict[str, Any],
    best_params_s: Dict[str, Any], output_dir: str = "output"
) -> int:
  """Function to run permutation classification with a single seed.

  Args:
      seed (int): Seed for reproducibility.
      spectras (np.ndarray): Spectras.
      file_names (np.ndarray): File names.
      sample_numbers (np.ndarray): Sample numbers.
      sample_types (np.ndarray): Sample types.
      who_grades (np.ndarray): WHO grades.
      model_type (str): The type of model to optimize hyperparameters for.
      best_params_r (Dict[str, Any]): Best hyperparameters for replica
          samples.
      best_params_s (Dict[str, Any]): Best hyperparameters for section
          samples.
      output_dir (str, optional): Output directory to save the results.

  Returns:
    int : Seed used for classification.

  """
  # Permute patient labels
  rng = np.random.default_rng(seed)
  unique_sample_numbers = np.unique(sample_numbers)
  sample_numbers_who_grades = np.array(
      [
          who_grades[sample_numbers == s_num][0]
          for s_num in unique_sample_numbers
      ]
  )
  permuted_sample_numbers_who_grades = rng.permutation(
      sample_numbers_who_grades
  )
  who_grades_permuted = np.array(
      [
          permuted_sample_numbers_who_grades[np.where(
              unique_sample_numbers == s_num
          )[0][0]] for s_num in sample_numbers
      ]
  )
  # Run classification
  single_seed_classification(
      seed=seed, spectras=spectras, file_names=file_names,
      sample_numbers=sample_numbers, sample_types=sample_types,
      who_grades=who_grades_permuted, model_type=model_type,
      best_params_r=best_params_r, best_params_s=best_params_s,
      output_dir=output_dir
  )
  # Create a folder for the seed
  seed_dir = output_dir / f"seed_{seed}"
  seed_dir.mkdir(parents=True, exist_ok=True)
  # Save the permuted sample numbers
  np.save(
      seed_dir / "permuted_sample_numbers_who_grades.npy",
      permuted_sample_numbers_who_grades
  )
  # Return the seed used for classification
  return seed


def permutation_classification_with_parallel(
    primary_seed: int, permutations: int, model_type: str,
    processed_files: List[Path], metadata_df: pd.DataFrame, model_dir: Path,
    output_dir: Path, n_jobs: int = -1
) -> List[int]:
  """Function to run permutation classification with multiple seeds in parallel.

  Args:
    primary_seed (int): Seed for reproducibility.
    permutations (int): Number of permutation to run.
    model_type (str): Type of model to use for classification.
    processed_files (List[Path]): List of processed files.
    metadata_df (pd.DataFrame): Metadata dataframe.
    model_dir (Path): Path to the directory containing the model saved results.
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
  # Generate multiple seeds for permutation
  permutation_seeds = [PRIMARY_SEED] + [
      int(i) for i in
      np.random.choice(range(10000), size=permutations - 1, replace=False)
  ]
  # Create the output directory if it does not exist
  output_dir.mkdir(parents=True, exist_ok=True)
  # Get the best parameters from the best seed
  best_params_r, best_params_s, _, _ = get_best_params_from_best_seed(
      model_dir, who_grades[sample_types == 'replica'],
      who_grades[sample_types == 'section']
  )
  # Use joblib to run the function in parallel for each seed
  results = [
      r for r in tqdm(
          Parallel(return_as="generator", n_jobs=n_jobs)(
              delayed(single_seed_permutation)(
                  seed=seed, spectras=spectras, file_names=file_names,
                  sample_numbers=sample_numbers, sample_types=sample_types,
                  who_grades=who_grades, best_params_r=best_params_r,
                  best_params_s=best_params_s, model_type=model_type,
                  output_dir=output_dir
              ) for seed in permutation_seeds
          ), total=len(permutation_seeds),
          desc="Running Permutation for multiple seeds"
      )
  ]
  # Return the list of seeds used for reference
  return permutation_seeds


def get_auc_from_best_seed(
    model_output_dir: Path, who_grades_r: np.ndarray, who_grades_s: np.ndarray
) -> Tuple[float, ...]:
  """ Function to get the auc from the best seed.

  Args:
    model_output_dir (Path): Path to the output directory containing model 
        saved results.
    who_grades_r (np.ndarray): WHO grades for non-bulk data (replica).
    who_grades_s (np.ndarray): WHO grades for non-bulk data (section).
  
  Returns:
    Tuple[float, ...]: Best auc for replica, section, replica-section, and
        section-replica.

  """
  # Initialize lists to store true labels and predictions
  y_true_r = (who_grades_r > 2).astype(int)
  y_true_s = (who_grades_s > 2).astype(int)
  # Define variables to store the best AUC scores
  best_auc_r, best_auc_s, best_auc_rs, best_auc_sr = (
      -np.inf, -np.inf, -np.inf, -np.inf
  )
  # Loop through each seed and load the saved predictions
  for seed_dir in model_output_dir.glob("seed_*"):
    if len(list(seed_dir.glob("*.npy"))) == 0:
      continue
    # Load predicted probabilities
    pred_r = np.load(seed_dir / "predicted_probabilities_r.npy")
    pred_s = np.load(seed_dir / "predicted_probabilities_s.npy")
    pred_rs = np.load(seed_dir / "predicted_probabilities_rs.npy")
    pred_sr = np.load(seed_dir / "predicted_probabilities_sr.npy")
    # Calculate ROC curves for each category
    auc_r = roc_auc_score(y_true_r, pred_r)
    auc_s = roc_auc_score(y_true_s, pred_s)
    # Check if the current model is better than the previous best model
    if auc_r > best_auc_r:
      best_auc_r = auc_r
      best_auc_rs = roc_auc_score(y_true_s, pred_rs)
    # Check if the current model is better than the previous best model
    if auc_s > best_auc_s:
      best_auc_s = auc_s
      best_auc_sr = roc_auc_score(y_true_r, pred_sr)
  return best_auc_r, best_auc_s, best_auc_rs, best_auc_sr


def get_permutations_aucs(
    permutation_output_path: Path, sample_numbers: np.ndarray,
    sample_types: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
  """Function to get the AUCs for all permutations.

  Args:
    permutation_output_path (Path): Path to the output directory containing 
      permutations saved results.
    sample_numbers (np.ndarray): Sample numbers.
    sample_types (np.ndarray): Sample types.

  Returns:
    Tuple[np.ndarray, np.ndarray]: AUCs for all 
        permutations for replica, section,

  """
  # Get the unique sample numbers
  unique_sample_numbers = np.unique(sample_numbers)
  # Define arrays to store interpolated TPR values for each seed
  aucs_r, aucs_s = [], []
  # Loop through each seed and load the saved predictions
  for seed_dir in permutation_output_path.glob("seed_*"):
    if len(list(seed_dir.glob("*.npy"))) == 0:
      continue
    # Load predicted probabilities
    pred_r = np.load(seed_dir / "predicted_probabilities_r.npy")
    pred_s = np.load(seed_dir / "predicted_probabilities_s.npy")

    # Load the permuted sample numbers and calculate the permuted WHO grades
    permuted_sample_numbers_who_grades = np.load(
        seed_dir / "permuted_sample_numbers_who_grades.npy"
    )
    who_grades_permuted = np.array(
        [
            permuted_sample_numbers_who_grades[np.where(
                unique_sample_numbers == s_num
            )[0][0]] for s_num in sample_numbers
        ]
    )
    # Get the permuted labels
    y_permuted = (who_grades_permuted > 2).astype(int)
    # Calculate ROC curves for each category
    aucs_r.append(roc_auc_score(y_permuted[sample_types == 'replica'], pred_r))
    aucs_s.append(roc_auc_score(y_permuted[sample_types == 'section'], pred_s))
  return np.array(aucs_r), np.array(aucs_s)


def plot_permutation_dist(
    permutation_auc_scores: np.ndarray, non_permuted_auc: float, ax: plt.Axes
) -> None:
  """
  Plot the distribution of permutation AUC scores and save the plot.

  Args:
    permutation_auc_scores (np.ndarray): Array of AUC scores from the 
        permutation test.
    non_permuted_auc (float): AUC score from the non-permuted data.
    ax (plt.Axes): Axis to plot the permutation test histogram.
  
  """
  # Calculate p-value
  p_value = (np.sum(permutation_auc_scores >= non_permuted_auc) +
             1) / (len(permutation_auc_scores) + 1)

  # Determine the p-value significance text
  if p_value < 0.0001:
    p_text = "* * * *"
  elif 0.0001 <= p_value < 0.001:
    p_text = "* * *"
  elif 0.001 <= p_value < 0.01:
    p_text = "* *"
  elif 0.01 <= p_value < 0.05:
    p_text = "*"
  else:
    p_text = "ns"
  # Plot the permutation test histogram
  ax = sns.histplot(
      permutation_auc_scores, bins=30, edgecolor='k', alpha=0.7,
      label='Permutation AUCs', ax=ax, color="tab:blue", stat="count"
  )
  # Plot the true AUC score
  ax.axvline(
      non_permuted_auc, color='red', linestyle='dashed',
      linewidth=fc.DEFAULT_LINE_WIDTH,
      label=f'True AUC ({non_permuted_auc:.3f})'
  )
  fc.set_titles_and_labels(ax, '', 'AUC Scores', 'Count')
  fc.customize_spines(ax)
  fc.customize_ticks(ax)
  # Add the legend
  l = ax.legend(
      loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2, fancybox=True,
      prop={"size": fc.DEFAULT_FONT_SIZE, "weight": fc.DEFAULT_FONT_WEIGHT}
  )
  for text in l.get_texts():
    text.set_color(fc.DEFAULT_COLOR)
  # Add p-value annotation above the red line
  ax.text(
      non_permuted_auc,
      ax.get_ylim()[1], p_text, ha='center', va='bottom', color='red',
      fontsize=fc.DEFAULT_FONT_SIZE, fontweight=fc.DEFAULT_FONT_WEIGHT
  )
  # Define the x-axis limits
  ax.set_xlim([-0.05, 1.05])
  return ax


def plot_roc_auc(
    model_output_dir: Path, who_grades_r: np.ndarray, who_grades_s: np.ndarray,
    sample_file_names_r: np.ndarray, sample_file_names_s: np.ndarray,
    ax: plt.Axes, agg: bool = False, cross_pred: bool = False
):
  """
  Function to plot ROC AUC curves.

  Args:
    model_output_dir (str): Path to the output directory containing model 
      saved results.
    who_grades_r (np.ndarray): WHO grades for non-bulk data (replica).
    who_grades_s (np.ndarray): WHO grades for non-bulk data (section).
    sample_file_names_r (np.ndarray): Sample file names for non-bulk data (replica).
    sample_file_names_s (np.ndarray): Sample file names for non-bulk data (section).
    ax (plt.Axes): Axis to plot the ROC AUC curves.
    agg (bool): Whether to aggregate the predictions by sample file names.
    cross_pred (bool): Whether to plot the ROC AUC curves for cross predictions.

  """
  # Prepare grouped indices for non-bulk data
  grouped_indices_r = prepare_grouped_indices(sample_file_names_r)
  grouped_indices_s = prepare_grouped_indices(sample_file_names_s)
  # Initialize lists to store true labels and predictions
  y_true_r = (who_grades_r > 2).astype(int)
  y_true_s = (who_grades_s > 2).astype(int)
  # Group the true labels by sample file names
  group_y_true_r = np.array(
      [y_true_r[indices][0] for indices in grouped_indices_r.values()]
  )
  group_y_true_s = np.array(
      [y_true_s[indices][0] for indices in grouped_indices_s.values()]
  )
  # Define arrays to store interpolated TPR values for each seed
  fpr_range = np.linspace(0, 1, 100)
  tprs_r, tprs_s = [], []
  aucs_r, aucs_s = [], []
  group_tprs_r, group_tprs_s = [], []
  group_aucs_r, group_aucs_s = [], []
  #
  seeds = []
  # Loop through each seed and load the saved predictions
  for seed_dir in model_output_dir.glob("seed_*"):
    if len(list(seed_dir.glob("*.npy"))) == 0:
      continue
    # Load predicted probabilities
    if not cross_pred:
      pred_r = np.load(seed_dir / "predicted_probabilities_r.npy")
      pred_s = np.load(seed_dir / "predicted_probabilities_s.npy")
    else:
      pred_r = np.load(seed_dir / "predicted_probabilities_sr.npy")
      pred_s = np.load(seed_dir / "predicted_probabilities_rs.npy")
    # Group the predictions by sample file names
    group_preds_r = np.array(
        [np.mean(pred_r[indices]) for indices in grouped_indices_r.values()]
    )
    group_preds_s = np.array(
        [np.mean(pred_s[indices]) for indices in grouped_indices_s.values()]
    )
    # Calculate ROC curves for each category
    fpr_r, tpr_r, _ = roc_curve(y_true_r, pred_r)
    fpr_s, tpr_s, _ = roc_curve(y_true_s, pred_s)
    group_fpr_r, group_tpr_r, _ = roc_curve(group_y_true_r, group_preds_r)
    group_fpr_s, group_tpr_s, _ = roc_curve(group_y_true_s, group_preds_s)
    # Calculate AUC scores directly using roc_auc_score
    aucs_r.append(roc_auc_score(y_true_r, pred_r))
    aucs_s.append(roc_auc_score(y_true_s, pred_s))
    group_aucs_r.append(roc_auc_score(group_y_true_r, group_preds_r))
    group_aucs_s.append(roc_auc_score(group_y_true_s, group_preds_s))
    # Interpolate TPR values to ensure consistency across different FPR values
    tprs_r.append(np.interp(fpr_range, fpr_r, tpr_r))
    tprs_s.append(np.interp(fpr_range, fpr_s, tpr_s))
    group_tprs_r.append(np.interp(fpr_range, group_fpr_r, group_tpr_r))
    group_tprs_s.append(np.interp(fpr_range, group_fpr_s, group_tpr_s))
    # Get the seed number
    seeds.append(int(seed_dir.stem.split("_")[-1]))
  # Calculate the mean and standard deviation of TPR values
  def calculate_mean_std(tprs, aucs):
    return np.mean(tprs, axis=0), np.std(tprs,
                                         axis=0), np.mean(aucs), np.std(aucs)

  # Calculate the mean and standard deviation of TPR values
  mean_tpr_r, std_tpr_r, mean_auc_r, std_auc_r = calculate_mean_std(
      tprs_r, aucs_r
  )
  mean_tpr_s, std_tpr_s, mean_auc_s, std_auc_s = calculate_mean_std(
      tprs_s, aucs_s
  )
  mean_group_tpr_r, std_group_tpr_r, mean_group_auc_r, std_group_auc_r = calculate_mean_std(
      group_tprs_r, group_aucs_r
  )
  mean_group_tpr_s, std_group_tpr_s, mean_group_auc_s, std_group_auc_s = calculate_mean_std(
      group_tprs_s, group_aucs_s
  )
  # Define the plot text
  if not cross_pred:
    plot_text_r = 'Replica'
    plot_text_s = 'Section'
  else:
    plot_text_r = 'Section-Replica'
    plot_text_s = 'Replica-Section'
  # Store the TPR values and AUC scores in dataframes
  tprs_r_df, tprs_s_df = None, None
  aucs_r_df, aucs_s_df = None, None
  # Plot the mean ROC curve with shaded standard deviation area
  if not agg:
    # Plot the mean ROC curve with shaded standard deviation area for non aggregated data
    ax.plot(
        fpr_range, mean_tpr_r, color='#F94040', lw=fc.DEFAULT_LINE_WIDTH,
        label=f'{plot_text_r} (AUC = {mean_auc_r:.2f} $\pm$ {std_auc_r:.2f})'
    )
    ax.fill_between(
        fpr_range, mean_tpr_r - std_tpr_r, mean_tpr_r + std_tpr_r,
        color='#F94040', alpha=0.25
    )
    ax.plot(
        fpr_range, mean_tpr_s, color='#5757F9', lw=fc.DEFAULT_LINE_WIDTH,
        label=f'{plot_text_s} (AUC = {mean_auc_s:.2f} $\pm$ {std_auc_s:.2f})'
    )
    ax.fill_between(
        fpr_range, mean_tpr_s - std_tpr_s, mean_tpr_s + std_tpr_s,
        color='#5757F9', alpha=0.25
    )
    tprs_r_df = pd.DataFrame(tprs_r, index=seeds, columns=fpr_range)
    tprs_s_df = pd.DataFrame(tprs_s, index=seeds, columns=fpr_range)
    aucs_r_df = pd.DataFrame({'AUC': aucs_r}, index=seeds)
    aucs_s_df = pd.DataFrame({'AUC': aucs_s}, index=seeds)
  else:
    # Plot the mean ROC curve with shaded standard deviation area for aggregated data
    ax.plot(
        fpr_range, mean_group_tpr_r, color='#F94040', lw=fc.DEFAULT_LINE_WIDTH,
        label=
        f'{plot_text_r} (AUC = {mean_group_auc_r:.2f} $\pm$ {std_group_auc_r:.2f})'
    )
    ax.fill_between(
        fpr_range, mean_group_tpr_r - std_group_tpr_r,
        mean_group_tpr_r + std_group_tpr_r, color='#F94040', alpha=0.25
    )
    ax.plot(
        fpr_range, mean_group_tpr_s, color='#5757F9', lw=fc.DEFAULT_LINE_WIDTH,
        label=
        f'{plot_text_s} (AUC = {mean_group_auc_s:.2f} $\pm$ {std_group_auc_s:.2f})'
    )
    ax.fill_between(
        fpr_range, mean_group_tpr_s - std_group_tpr_s,
        mean_group_tpr_s + std_group_tpr_s, color='#5757F9', alpha=0.25
    )
    tprs_r_df = pd.DataFrame(group_tprs_r, index=fpr_range, columns=seeds)
    tprs_s_df = pd.DataFrame(group_tprs_s, index=fpr_range, columns=seeds)
    aucs_r_df = pd.DataFrame({'AUC': group_aucs_r}, index=seeds)
    aucs_s_df = pd.DataFrame({'AUC': group_aucs_s}, index=seeds)

  # Plot the random classifier line
  ax.plot(
      [0, 1], [0, 1], color=fc.DEFAULT_COLOR, lw=fc.DEFAULT_LINE_WIDTH,
      linestyle='--'
  )
  # Set the axis limits
  ax.set_xlim([-0.01, 1.01])
  ax.set_ylim([-0.01, 1.01])
  # Set the aspect ratio
  ax.set_aspect('equal', adjustable='box')
  # Customize the plot
  fc.set_titles_and_labels(ax, '', 'FPR', 'TPR')
  fc.customize_spines(ax, linewidth=3.5)
  fc.customize_ticks(ax)
  ax.tick_params(axis='both', length=10, width=3.5)
  # Add the legend
  l = ax.legend(
      loc='lower right', prop={"size": 10, 'weight': fc.DEFAULT_FONT_WEIGHT}
  )
  for text in l.get_texts():
    text.set_color(fc.DEFAULT_COLOR)
  return ax, tprs_r_df, tprs_s_df, aucs_r_df, aucs_s_df


def perform_loocv_shap(
    X: np.ndarray, y: np.ndarray, batch_ids: np.ndarray,
    patient_ids: np.ndarray, model_type: str, seed: int,
    best_params_list: List[Dict[str, Any]] = None
) -> None:
  # Initialize the SHAP values
  shap_values_all = np.zeros((X.shape))
  # Prepare grouped indices for LOOCV
  grouped_indices = prepare_grouped_indices(batch_ids)
  # Loop over each group for training and testing
  for idx, (_, test_idx) in tqdm(
      enumerate(grouped_indices.items()), total=len(grouped_indices),
      desc="LOOCV"
  ):
    # Define train indices by excluding patients in the test set
    train_idx = ~np.isin(patient_ids, patient_ids[test_idx])
    # Get the best parameters if provided, otherwise train and optimize
    best_params = best_params_list[idx] if best_params_list else None
    # Create and fit the model using the best parameters
    best_model = create_best_model(model_type, best_params, seed)
    best_model.fit(X[train_idx], y[train_idx])
    # Use SHAP to explain the prediction for the left-out sample
    if isinstance(
        best_model, (RandomForestClassifier, XGBClassifier, LGBMClassifier)
    ):
      explainer = shap.TreeExplainer(best_model, X[train_idx])
    else:
      explainer = shap.Explainer(best_model, X[train_idx])
    shap_values = explainer(X[test_idx])
    if shap_values.values.ndim == 3:  # Multiclass case
      # Example: Using SHAP values for the first class
      shap_values_all[test_idx, :] = shap_values.values[:, :, 1]
    else:
      shap_values_all[test_idx, :] = shap_values.values
    return shap_values_all


def plot_shap_explanations(
    spectras: np.ndarray, who_grades: np.ndarray, file_names: np.ndarray,
    sample_numbers: np.ndarray, sample_types: np.ndarray, model_type: str,
    processed_files: List[Path], figures_path: Path
) -> None:
  # Separate the data based on the sample type
  (X_r, X_s, y_r, y_s, batch_ids_r, batch_ids_s, patient_ids_r,
   patient_ids_s) = separate_data_by_sample_type(
       spectras, (who_grades > 2).astype(int), file_names, sample_numbers,
       sample_types
   )
  # Read the common m/z values
  features = np.load(processed_files[0] / "common_mzs.npy")
  # Get the best parameters from the best seed
  best_params_r, best_params_s, seed_r, seed_s = get_best_params_from_best_seed(
      figures_path / model_type, who_grades[sample_types == 'replica'],
      who_grades[sample_types == 'section']
  )
  # Perform LOOCV SHAP for each sample type
  shap_values_r = perform_loocv_shap(
      X_r, y_r, batch_ids_r, patient_ids_r, model_type, seed_r, best_params_r
  )
  shap_values_s = perform_loocv_shap(
      X_s, y_s, batch_ids_s, patient_ids_s, model_type, seed_s, best_params_s
  )
  # Plot the SHAP explanations for each sample type
  for shap_values, X, sample_type in zip(
      [shap_values_r, shap_values_s], [X_r, X_s], ["Replica", "Section"]
  ):
    # Plot the SHAP summary plot
    shap.plots.beeswarm(
        shap.Explanation(
            values=shap_values, data=X,
            feature_names=[str(i) for i in features]
        ), show=False, max_display=20
    )
    # Retrieve the current figure
    fig, ax = plt.gcf(), plt.gca()
    # Set the figure size
    fig.set_size_inches(3, 9)
    # Get current y-axis labels
    labels = [label.get_text() for label in plt.gca().get_yticklabels()]
    labels[1:] = [f"{float(label):.4f}" for label in labels[1:]]
    labels[0] = f"Σ{len(features) - 20} features"
    ax.set_yticklabels(labels)
    # Customize the plot
    fc.set_titles_and_labels(ax, '', 'SHAP value', '')
    fc.customize_ticks(ax)
    # fc.customize_colorbar(fig.colorbar)
    ax.spines["bottom"].set_linewidth(fc.DEFAULT_LINE_WIDTH)
    ax.spines["bottom"].set_color(fc.DEFAULT_COLOR)
    # Customize the colorbar
    cbar = None
    for axis in fig.axes:
      if hasattr(axis, 'collections') and axis.collections:
        cbar = axis
    cbar.set_ylabel(
        'Feature value', fontsize=fc.DEFAULT_FONT_SIZE,
        fontweight=fc.DEFAULT_FONT_WEIGHT, color=fc.DEFAULT_COLOR
    )
    for label in cbar.get_yticklabels():
      label.set_fontweight(fc.DEFAULT_FONT_WEIGHT)
      label.set_fontsize(fc.DEFAULT_FONT_SIZE)
      label.set_color(fc.DEFAULT_COLOR)
    for spine in cbar.spines.values():
      spine.set_linewidth(fc.DEFAULT_LINE_WIDTH)
      spine.set_color(fc.DEFAULT_COLOR)
    # Save the plot
    plt.savefig(
        figures_path / f"shap_beeswarm_{sample_type.lower()}.png",
        bbox_inches='tight', dpi=1200, transparent=True
    )
    plt.show()
    plt.close()
    # Plot the SHAP bar plot
    shap.plots.bar(
        shap.Explanation(
            values=shap_values, feature_names=[str(i) for i in features]
        ), show=False, max_display=20
    )
    # Retrieve the current figure
    fig, ax = plt.gcf(), plt.gca()
    # Set the figure size
    fig.set_size_inches(3, 9)
    # Get current y-axis labels
    labels = [label.get_text() for label in ax.get_yticklabels()]
    labels[:19] = [f"{float(label):.4f}" for label in labels[:19]]
    labels[20:-1] = [f"{float(label):.4f}" for label in labels[20:-1]]
    labels[19] = f"Σ{len(features) - 20} features"
    labels[-1] = f"Σ{len(features) - 20} features"
    ax.set_yticklabels(labels)
    # Customize the plot
    fc.set_titles_and_labels(ax, '', 'mean(|SHAP value|)', '')
    fc.customize_ticks(ax, rotate_x_ticks=45)
    fc.customize_spines(ax)
    plt.savefig(
        figures_path / f"shap_bar_{sample_type.lower()}.png",
        bbox_inches='tight', dpi=1200, transparent=True
    )
    plt.close()
    plt.show()


def plot_and_save_figures(
    metadata_df: pd.DataFrame, figures_path: Path, processed_files: List[Path],
    model_type: str
):
  # Load the data
  (
      spectras, file_names, sample_file_names, sample_numbers, sample_types,
      who_grades
  ) = load_data(processed_files, metadata_df)
  # Define the paths
  model_output_dir = figures_path / model_type
  permutation_output_path = figures_path / "permutation" / model_type
  # Get best auc from the best seed
  best_auc_r, best_auc_s, best_auc_rs, best_auc_sr = get_auc_from_best_seed(
      model_output_dir, who_grades[sample_types == "replica"],
      who_grades[sample_types == "section"]
  )
  # Get the permutation AUCs
  permutation_aucs_r, permutation_aucs_s = get_permutations_aucs(
      permutation_output_path, sample_numbers, sample_types
  )
  # Plot the permutation test distribution
  for aucs, best_auc, sample_type in zip(
      [permutation_aucs_r, permutation_aucs_s], [best_auc_r, best_auc_s],
      ["r", "s"]
  ):
    fig, ax = plt.subplots(1, figsize=(7.79, 5.51))
    plot_permutation_dist(aucs, best_auc, ax)
    plt.tight_layout()
    plt.savefig(
        figures_path / f"permutation_test_{sample_type}.png",
        bbox_inches='tight', dpi=1200, transparent=True
    )
    plt.show()
    plt.close()
  # Plot the roc auc
  fig, ax = plt.subplots(1, figsize=(5.845 * 1.25, 4.135 * 1.25))
  ax, tprs_r_df, tprs_s_df, aucs_r_df, aucs_s_df = plot_roc_auc(
      model_output_dir, who_grades[sample_types == "replica"],
      who_grades[sample_types == "section"],
      sample_file_names[sample_types == "replica"],
      sample_file_names[sample_types == "section"], ax, agg=False,
      cross_pred=False
  )
  plt.tight_layout()
  plt.savefig(
      figures_path / "roc_auc_spectra_wise.png", bbox_inches='tight', dpi=1200,
      transparent=True
  )
  plt.show()
  plt.close()
  # Plot the cross modality roc auc
  fig, ax = plt.subplots(1, figsize=(5.845 * 1.25, 4.135 * 1.25))
  ax, tprs_sr_df, tprs_rs_df, aucs_sr_df, aucs_rs_df = plot_roc_auc(
      model_output_dir, who_grades[sample_types == "replica"],
      who_grades[sample_types == "section"],
      sample_file_names[sample_types == "replica"],
      sample_file_names[sample_types == "section"], ax, agg=False,
      cross_pred=True
  )
  plt.tight_layout()
  plt.savefig(
      figures_path / "roc_auc_spectra_wise_cross_modality.png",
      bbox_inches='tight', dpi=1200, transparent=True
  )
  plt.show()
  plt.close()
  # Plot the SHAP explanations
  plot_shap_explanations(
      spectras, who_grades, file_names, sample_numbers, sample_types,
      model_type, processed_files, figures_path
  )
  # Save the data to create the figures
  pd.DataFrame({'AUC': permutation_aucs_r}).to_csv(
      figures_path / f"permutation_aucs_r_best_real_auc_{best_auc_r:.4f}.csv"
  )
  pd.DataFrame({'AUC': permutation_aucs_s}).to_csv(
      figures_path / f"permutation_aucs_s_best_real_auc_{best_auc_s:.4f}.csv"
  )
  tprs_r_df.to_csv(figures_path / "tprs_r_df.csv")
  tprs_s_df.to_csv(figures_path / "tprs_s_df.csv")
  aucs_r_df.to_csv(figures_path / "aucs_r_df.csv")
  aucs_s_df.to_csv(figures_path / "aucs_s_df.csv")
  tprs_rs_df.to_csv(figures_path / "tprs_rs_df.csv")
  tprs_sr_df.to_csv(figures_path / "tprs_sr_df.csv")
  aucs_rs_df.to_csv(figures_path / "aucs_rs_df.csv")
  aucs_sr_df.to_csv(figures_path / "aucs_sr_df.csv")


def main(
    figures_path: Path, processed_files: List[Path], model_type: str,
    n_iterations: int, n_permutations: int
):
  """Function containing main code"""
  # Define the output path
  output_path = figures_path / model_type
  output_path.mkdir(parents=True, exist_ok=True)
  # Run the classification with multiple seeds in parallel
  evaluation_seeds = multiple_seeds_classification_with_parallel(
      PRIMARY_SEED, n_iterations, model_type, processed_files, metadata_df,
      output_path, n_trials=50, n_jobs=-1
  )
  # Run the permutation test
  permutation_output_path = figures_path / "permutation" / model_type
  permutation_classification_with_parallel(
      PRIMARY_SEED, n_permutations, model_type, processed_files, metadata_df,
      output_path, permutation_output_path, n_jobs=-1
  )
  # Create the figures dir
  if not figures_path.exists():
    figures_path.mkdir(parents=True, exist_ok=True)
  # Plot and save the figures
  plot_and_save_figures(metadata_df, figures_path, processed_files, model_type)


# Define current folder using this file
CWD = Path(os.path.dirname(os.path.abspath(__file__)))
# Define folder that contains dataset
DHG_PATH = CWD / ".." / ".." / "data" / "DHG"
# Define folder that contains raw data
RAW_DATA = DHG_PATH / "raw_txt"
# Define folder to save processed data
PROCESSED_DATA = DHG_PATH / "processed_txt"
# Define file that contains metadata
METADATA_PATH = DHG_PATH / "txt_metadata.csv"
# Define the primary seed for reproducibility
PRIMARY_SEED = 42
# Read metadata csv
metadata_df = pd.read_csv(METADATA_PATH)

if __name__ == "__main__":
  # Define path to save plots and results
  FIGURES_PATH = CWD / "dhg" / "classification"
  FIGURES_PATH.mkdir(exist_ok=True, parents=True)
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
      '--model_type', type=str,
      choices=['logistic_regression', 'random_forest', 'lightgbm',
               'xgboost'], default='lightgbm', help=(
                   "Type of model to use (e.g., 'logistic',"
                   "'random_forest', 'lightgbm' or 'xgboost')"
               )
  )
  args = parser.parse_args()
  # Get the processed files
  processed_files = list(Path(PROCESSED_DATA).iterdir())
  # Run the main function
  main(
      FIGURES_PATH, processed_files, args.model_type, args.n_iterations,
      args.n_permutations
  )
