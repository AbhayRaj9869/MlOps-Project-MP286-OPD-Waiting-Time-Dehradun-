from __future__ import annotations
from pathlib import Path
import yaml
from .quality import findings as run_findings
from .config import ROOT

def write_data_findings(path=None):
    out = run_findings()
    dest = Path(path or ROOT/"reports/findings_data.yaml")
    dest.parent.mkdir(exist_ok=True); dest.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return out
