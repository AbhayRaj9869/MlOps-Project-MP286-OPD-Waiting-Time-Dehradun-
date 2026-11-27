from __future__ import annotations

# Import Path for handling filesystem paths in a platform-independent way.
from pathlib import Path

# Import YAML for writing the findings into a YAML-formatted file.
import yaml

# Import the findings function from the quality module and rename it to run_findings.
from .quality import findings as run_findings

# Import ROOT, which represents the project's root directory.
from .config import ROOT


def write_data_findings(path=None):
    # Run the data quality inspection function to generate all findings.
    out = run_findings()

    # Define the output file path.
    # If a custom path is provided, use it; otherwise, save the report
    # inside the project's reports directory as findings_data.yaml.
    dest = Path(path or ROOT/"reports/findings_data.yaml")

    # Create the parent directory if it does not already exist.
    # exist_ok=True prevents an error if the directory already exists.
    dest.parent.mkdir(exist_ok=True)

    # Convert the findings dictionary into YAML format and write it to the file.
    # sort_keys=False preserves the original order of the dictionary keys.
    # encoding="utf-8" ensures proper text encoding.
    dest.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")

    # Return the findings dictionary so it can also be used by other functions
    # without needing to read the generated YAML file again.
    return out
