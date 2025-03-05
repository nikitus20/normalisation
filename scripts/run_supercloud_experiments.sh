#!/bin/bash
# Script to run all normalization experiments on MIT Supercloud

# Create logs directory
mkdir -p slurm_logs

# Function to submit an experiment
submit_experiment() {
    CONFIG_PATH=$1
    EXPERIMENT_NAME=$(basename "$CONFIG_PATH" .yaml)
    
    echo "Submitting experiment: $EXPERIMENT_NAME"
    JOBID=$(sbatch --parsable run_experiment.slurm "$CONFIG_PATH")
    
    echo "Job submitted with ID: $JOBID"
    echo "$EXPERIMENT_NAME: $JOBID" >> experiment_jobs.log
}

# Clear previous log
echo "# Experiment Jobs" > experiment_jobs.log
echo "Started at: $(date)" >> experiment_jobs.log
echo "----------------------------------------" >> experiment_jobs.log

# Submit experiments for different normalization approaches
echo "Submitting normalization experiments..."

# 1. PreLN experiment
submit_experiment "configs/experiments/preln.yaml"

# 2. PostLN experiment
submit_experiment "configs/experiments/postln.yaml"

# 3. Custom normalization experiment
submit_experiment "configs/experiments/custom_norm.yaml"

# 4. Warmup experiments
submit_experiment "configs/training/warmup_experiments.yaml"

echo "All experiments submitted!"
echo "Use 'squeue -u \$USER' to check status"
echo "Experiment IDs saved to experiment_jobs.log"