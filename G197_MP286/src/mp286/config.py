
from pathlib import Path
import yaml


# ---------------------------------------------------------
# Project Configuration
# ---------------------------------------------------------

# Resolve the project root directory.
# parents[2] moves two levels up from the current file.
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------

def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    """
    Load project configuration parameters from a YAML file.

    Args:
        path: Location of the configuration file.
              Defaults to params.yaml in the project root.

    Returns:
        dict: Configuration parameters loaded from YAML.
    """

    with open(path, encoding="utf-8") as f:
        params = yaml.safe_load(f)

    return params
