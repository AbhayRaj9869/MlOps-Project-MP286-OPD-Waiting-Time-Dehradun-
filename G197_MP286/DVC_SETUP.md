# DVC setup for Group G197

The three `.dvc` pointer files are stored beside the original data under `data/raw/`. The raw CSV files are ignored by Git, as required by the handbook.

On the machine that has the original CSVs:

```powershell
python -m pip install -r requirements.txt
dvc status
```

Before a clean clone can download the datasets, the group must configure one shared DVC remote that all members can access. Use the storage URL and credentials agreed by the group/faculty; do not commit credentials. Then run:

```powershell
dvc remote add -d group-storage <SHARED_DVC_REMOTE_URL>
dvc push
```

Each teammate configures the same remote (credentials stay local) and runs `dvc pull`. A remote URL cannot be safely invented: it depends on the group's chosen storage provider and account.
