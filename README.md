# Latent Reasoning DeepSeek-R1

A framework for implementing latent reasoning techniques in mathematical problem-solving using large language models (LLMs).

## Core Components

## Setup and Usage

1. Environment setup:
   ```bash
   conda env create -f src/environment.yml
   ```

### Main Training and Inference
- `main.py` - Primary entry point for GSM8K dataset training and evaluation
  - Coordinates model training, LoRA adaptation, and latent reasoning generation
  - Uses GSM8K dataset for mathematical problem solving

- `jai_main.py` - Entry point for combinatorics/textbook problem training
  - Similar to main.py but specialized for theoretical math problems
  - Uses custom textbook datasets

### Core Implementation
- `latent_reasoning.py` - Core latent reasoning implementation
  - Implements latent space reasoning techniques
  - Provides batch and single-sample generation methods
  - Handles model inference with latent space manipulation

- `train.py` - Training loop implementation
  - Implements the core training logic
  - Handles optimization and learning rate scheduling

### Data Handling
- `utils.py` - Core utilities and data processing
  - Provides data formatting and collation functions
  - Implements dataset iteration utilities
  - Contains prompt formatting for different problem types

- `gsm8k.py` - GSM8K dataset handler
  - Loads and processes the GSM8K math dataset
  - Creates data loaders for training

- `textbook_data.py` - Theoretical math problems handler
  - Processes combinatorics and convex optimization problems
  - Creates specialized data loaders for theoretical math

### Model Adaptation and Experimentation
- `lora.py` - Low-Rank Adaptation implementation
  - Applies LoRA for efficient model fine-tuning
  - Manages parameter-efficient training

- `experiments.py` - Experimental framework
  - Implements evaluation metrics
  - Provides utilities for running experiments
  - Handles model prediction and accuracy assessment

### Job Management
- `main_job.py` - Job-specific training entry point
  - Handles command-line arguments for training jobs
  - Configures training parameters

- `main_scheduler.py` - Training job scheduler
  - Manages multiple training runs
  - Handles job scheduling and resource allocation

## Dependencies

The project primarily depends on:
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- NumPy
- scikit-learn

## Usage

1. For GSM8K math problem training:
```bash
python main.py
```

2. For theoretical math problems:
```bash
python jai_main.py
```

3. For distributed training jobs:
```bash
python main_scheduler.py
```

## Project Structure

```
src/
├── datasets/              # Dataset files
│   ├── combinatorics.json # Combinatorics problems dataset
│   └── convex.json       # Convex optimization problems dataset
├── jobs/                 # Job configuration files
├── utils.py             # Utility functions for data processing and model training
├── main.py              # Main training script
├── train.py             # Core training logic
├── gsm8k.py            # GSM8K dataset handling
├── latent_reasoning.py  # Latent reasoning implementation
├── lora.py             # LoRA (Low-Rank Adaptation) implementation
├── experiments.py      # Experimental configurations
```

## Key Components

- **Latent Reasoning**: Implementation of latent space reasoning techniques for mathematical problem-solving (`latent_reasoning.py`)
- **Data Processing**: Utilities for handling various mathematical datasets including GSM8K and custom datasets (`utils.py`, `gsm8k.py`)
- **Training**: Core training logic with support for different optimization strategies (`train.py`)
- **Experimentation**: Configurable experiment framework with analysis tools (`experiments.py`, `analyze_results.py`)

## Datasets

The project uses multiple mathematical problem datasets:
- GSM8K dataset (handled through `gsm8k.py`)
- Theoretical math problems (handled through `textbook_data.py`):
  - Combinatorics problems (`datasets/combinatorics.json`)
  - Convex optimization problems (`datasets/convex.json`)

## Analysis Tools

The repository includes tools for analyzing model performance and behavior:
- Take a pretrained model and test on GSM8K (`experiments.py`)
