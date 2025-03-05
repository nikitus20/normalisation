#!/bin/bash
#SBATCH --job-name=llama_norm
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load required modules for MIT Supercloud
module load anaconda/2023a
module load cuda/11.8

# Activate environment
source activate norm

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Running experiment on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
echo "Configuration file: $1"

# Run experiment with provided config file
python -m src.experiments.run_experiment --config $1

# Print completion message
echo "Experiment completed at $(date)"