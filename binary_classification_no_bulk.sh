#!/bin/bash

#SBATCH --partition main			           ### specify partition name where to run a job
#SBATCH --time 7-00:00:00			           ### limit the time of job running
#SBATCH --job-name slgc_binary_classification		       ### name of the job
#SBATCH --output job-%A_%a.out			       ### output log for running job - %A for array job ID, %a for array index
#SBATCH --mail-user=leorro@post.bgu.ac.il    ### user's email for sending job status messages
#SBATCH --mail-type=ALL			           ### conditions for sending the email
#SBATCH --mem=128G				               ### amount of RAM memory
#SBATCH --cpus-per-task=8			       ### number of CPU cores per task
#SBATCH --array=3		           ### define job array with indices

### Array of Python commands
commands=(
  "python -u -m binary_classification_no_bulk --n_iterations 100 --n_permutations 1000 --model_type logistic_regression"
  "python -u -m binary_classification_no_bulk --n_iterations 100 --n_permutations 1000 --model_type decision_tree"
  "python -u -m binary_classification_no_bulk --n_iterations 100 --n_permutations 1000 --model_type random_forest"
  "python -u -m binary_classification_no_bulk --n_iterations 100 --n_permutations 1000 --model_type lightgbm"
  "python -u -m binary_classification_no_bulk --n_iterations 100 --n_permutations 1000 --model_type xgboost"
)

### Load modules and activate conda environment
module load anaconda
source activate tfgpu_jup

### Change to the working directory
cd "/sise/assafzar-group/assafzar/leor/delta_tissue_slgc/Nondestructive Spatial Lipidomics for Glioma Classification - Tissue Similarity and Grading/SLGC-tissue-similarity-and-grading/"

### Run the command corresponding to the array index
eval ${commands[$SLURM_ARRAY_TASK_ID]}