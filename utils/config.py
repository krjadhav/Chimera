"""
Configuration utilities for ChimeraAI.

This module provides functions for loading and managing configuration files.
"""

import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    The override_config values take precedence over base_config values.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged_config = base_config.copy()
    
    def _merge_dicts(base: Dict, override: Dict) -> Dict:
        """Recursively merge nested dictionaries."""
        for key, value in override.items():
            if (
                key in base and 
                isinstance(base[key], dict) and 
                isinstance(value, dict)
            ):
                base[key] = _merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
    
    return _merge_dicts(merged_config, override_config)


def get_default_config_path(config_name: str) -> str:
    """
    Get the path to a default configuration file.
    
    Args:
        config_name: Name of the configuration file
        
    Returns:
        Path to the configuration file
    """
    # Get the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Construct path to the configs directory
    configs_dir = os.path.join(root_dir, 'configs')
    
    # Add .yaml extension if not already present
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'
    
    # Construct the full path
    config_path = os.path.join(configs_dir, config_name)
    
    return config_path
