"""
Config file utility for loading YAML configuration.
Supports optional configuration for hitl-autofe_v3.py
"""

import yaml
import os
from pathlib import Path

def load_config(config_path="config.yml"):
    """
    Load configuration from YAML file.
    
    Parameters:
    config_path (str): Path to config.yml file. Default: config.yml in current directory.
    
    Returns:
    dict: Configuration dictionary with keys:
        - data_path: Path to experiment data
        - steps: Number of optimization steps
        - llm_model: LLM model name
        - drop_threshold: Threshold for acceptable performance drop
        - temperature: LLM temperature parameter
        - dir_name: Directory name for results (instead of timestamp)
        - seeds: List of seeds to process
        - masked: Whether to mask feature names
        - processed_datasets: Datasets to skip
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_config_or_default(config_path=None):
    """
    Load config if provided, otherwise return None.
    Allows graceful fallback to hardcoded defaults in main code.
    
    Parameters:
    config_path (str or None): Path to config.yml. If None, returns None.
    
    Returns:
    dict or None: Configuration dictionary, or None if config_path is None.
    """
    if config_path is None:
        return None
    
    return load_config(config_path)

def validate_config(config):
    """
    Validate that all required keys are present in config.
    
    Parameters:
    config (dict): Configuration dictionary
    
    Returns:
    bool: True if valid
    
    Raises:
    ValueError: If required keys are missing
    """
    required_keys = ['data_path', 'steps', 'llm_model', 'drop_threshold', 'temperature']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    return True

def create_sample_config(output_path="config_sample.yml"):
    """
    Create a sample config.yml file for reference.
    
    Parameters:
    output_path (str): Where to save the sample config file
    """
    sample_config = """# Configuration for hitl-autofe_v3.py

# Data configuration
data_path: /home/xylar369/experiment_data

# Optimization settings
steps: 50

# LLM settings
llm_model: neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8
temperature: 0.2

# Feature engineering settings
drop_threshold: 1.5  # Threshold for acceptable performance drop

# Output directory name (instead of timestamp)
# Will be used as: /home/xylar369/results/hitl_autofe/{llm_model_name}/{dir_name}/results
dir_name: experiment_run_v1

# Seeds to process
seeds:
  - "0"
  - "1"
  - "2"
  - "3"
  - "4"

# Data masking
masked: true

# Datasets to skip (already processed)
processed_datasets:
  - airlines
  - jungle_chess_2pcs_raw_endgame_complete
  - pc1
"""
    
    with open(output_path, 'w') as f:
        f.write(sample_config)
    
    print(f"Sample config created at: {output_path}")

def save_config(config, output_dir, filename="config_used.yml"):
    """
    Save the current configuration to the results directory.
    Helps identify which experiment configuration was used.
    
    Parameters:
    config (dict): Configuration dictionary to save
    output_dir (str): Directory where to save the config file
    filename (str): Name of the output file (default: config_used.yml)
    
    Returns:
    str: Path to the saved config file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    # Convert config dict to YAML and save
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Config saved to: {output_path}")
    return output_path

def save_config_and_defaults(config_dict, output_dir, filename="config_used.yml"):
    """
    Save configuration including both loaded config and default values.
    Makes it very clear what parameters were used for the experiment.
    
    Parameters:
    config_dict (dict): Configuration to save
    output_dir (str): Directory where to save the config file
    filename (str): Name of the output file
    
    Returns:
    str: Path to the saved config file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    # Add metadata about when this was saved
    from datetime import datetime
    config_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'config': config_dict
    }
    
    # Convert to YAML and save
    with open(output_path, 'w') as f:
        yaml.dump(config_with_metadata, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Config with metadata saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Example usage
    create_sample_config()
    print("Sample config created. Edit config.yml and run with config loading enabled.")
