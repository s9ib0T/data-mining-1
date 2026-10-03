# data-mining-1

Using the Copenhagen Networks Study dataset, we aim to answer the following:
> Can you tell who someone's friends are just from which phones are near theirs?

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
scripts/download.sh
```

When opening a notebook, select `.venv/bin/python` as the kernel; the notebooks need the packages installed in `.venv`.

`scripts/download.sh` downloads the Copenhagen Networks Study data from figshare to `data/copenhagen/`. It skips files that already exist. The first code cell in `notebooks/preprocess.ipynb` runs the script.

Do not edit the files in `data/copenhagen/`. Save changed data to `data/processed/`.

Analysis code is in `src/`, and the notebooks import it.

---

`notebooks/preprocess.ipynb` cleans the raw data and saves to `data/processed/`. `friends.ipynb` and `isolated.ipynb` load data with `proximity()` and `student_table()` from `src/preprocess.py`. No need for `preprocess.ipynb` to run first, both funcs builds `data/processed/` from the raw files if needed. Deleting `processed/` is safe.

`friends.ipynb` answers the question "Can you tell who someone's friends are just from which phones are near theirs?"
