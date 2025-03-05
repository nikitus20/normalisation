# LLaMA-71M Normalization Research

This project implements a custom 71M parameter LLaMA model to experiment with different normalization approaches:
- **PreLN**: Pre-Layer Normalization, placing the normalization at the beginning of each transformer sub-block
- **PostLN**: Post-Layer Normalization, placing the normalization at the end of each transformer sub-block
- **Custom Normalization**: Experimenting with alternative normalization techniques and placements

The project provides a flexible framework to configure models, training procedures, and compare different normalization strategies with a focus on implementation control.

## Project Features

- Customizable model implementations (LLaMA-71M, BERT, NanoGPT)
- Flexible normalization strategies (LayerNorm, RMSNorm, custom approaches)
- Random data warmup capabilities
- Comprehensive experiment configuration system
- Performance metrics tracking and comparison
- Support for local and remote (MIT Supercloud) execution

## Project Structure

```
├── configs/                # Configuration files
│   ├── experiments/        # Experiment-specific configurations
│   ├── models/             # Model architecture configurations
│   └── training/           # Training procedure configurations
├── scripts/                # Utility scripts
│   └── download_weights.sh # Script to download pretrained weights
├── src/                    # Source code
│   ├── data/               # Data loading and preprocessing
│   ├── experiments/        # Experiment running logic
│   ├── models/             # Model implementations
│   │   └── layers/         # Model components (attention, FFN, normalization)
│   ├── training/           # Training utilities
│   └── utils/              # Common utilities
└── README.md               # This file
```

## Setup and Dependencies

### Local Setup

1. Create a virtual environment and install dependencies:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. Download model weights (optional):

```bash
bash scripts/download_weights.sh
```

## Running Experiments

### Single Experiment

Run one of the predefined normalization experiments:

```bash
# Run PreLN experiment
python -m src.experiments.run_experiment --config configs/experiments/preln.yaml

# Run PostLN experiment
python -m src.experiments.run_experiment --config configs/experiments/postln.yaml

# Run custom normalization experiment
python -m src.experiments.run_experiment --config configs/experiments/custom_norm.yaml
```

### Warmup with Random Data

Experiment with random data warmup:

```bash
# Run experiment with random data warmup
python -m src.experiments.run_experiment --config configs/training/warmup_experiments.yaml
```

### Compare All Normalization Approaches

Run and compare all normalization strategies:

```bash
python -m src.experiments.run_experiment --compare
```

### Customizing Experiments

You can override specific configuration parameters:

```bash
python -m src.experiments.run_experiment --config configs/experiments/preln.yaml --learning_rate 1e-4 --warmup_steps 1000
```

## Running on MIT Supercloud

To run experiments on MIT Supercloud:

1. Connect to the Supercloud cluster:

```bash
ssh <your-username>@txe1-login.mit.edu
```

2. Set up your environment:

```bash
# Create a project directory
mkdir -p ~/projects
cd ~/projects

# Clone the repository
git clone <repository-url> normalisation
cd normalisation

# Load required modules
module load anaconda/2023a
module load cuda/11.8

# Create and activate conda environment
conda create -n norm python=3.10
conda activate norm

# Install dependencies
pip install -r requirements.txt
```

3. Create a SLURM job script (save as `run_experiment.slurm`):

```bash
#!/bin/bash
#SBATCH --job-name=llama_norm
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Load modules
module load anaconda/2023a
module load cuda/11.8

# Activate environment
source activate norm

# Run experiment
python -m src.experiments.run_experiment --config $1
```

4. Create logs directory:

```bash
mkdir -p slurm_logs
```

5. Submit a job:

```bash
sbatch run_experiment.slurm configs/experiments/preln.yaml
```

6. Monitor job status:

```bash
squeue -u <your-username>
```

## Extending the Project

### Adding a New Normalization Technique

1. Modify `src/models/layers/normalization.py` to add your custom normalization:
   - Add your implementation to the `CustomNorm` class or create a new class
   - Update the `NormFactory` to include your new normalization type

2. Create a new configuration in `configs/experiments/` for your experiment

### Adding a New Model Architecture

1. Implement your model in `src/models/`
2. Create a configuration in `configs/models/`
3. Update the `create_model` method in `src/experiments/run_experiment.py`

## Results and Output

Experiment results are saved in the `outputs/` directory with the following structure:

- `outputs/<experiment_name>/`:
  - `experiment_config.yaml`: Copy of experiment configuration
  - `model/`: Saved model checkpoints
  - `results.json`: Performance metrics
  - `logs/`: Training logs

## License

MIT License