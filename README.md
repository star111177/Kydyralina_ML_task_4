# Task 4 — Random Forests & Ensembles

Starter files for **Task 4** (graded, 25 points, due Sunday 15 November 2026, 23:59 Canvas time, UTC+5)
of *Machine Learning (Python)*, MOP 3231, Narxoz University.

The task text, the six parts, what to submit and the grading table are in the handout in the **Week 08** module
in Canvas. This repository holds only the files you start from; it has no instructions of its own beyond the setup
below.

## What is here

| File | What it is |
|---|---|
| `data/spambase.csv` | The dataset: 4,204 e-mails described by 57 numeric features, and the label `spam` (1 = spam, 0 = legitimate mail) |
| `requirements.txt` | Pinned library versions the task is written and checked against. Do not change them |
| `make_dataset.py` | How `spambase.csv` was built from the UCI source. You do not need to run it |
| `.gitignore` | Keeps your virtual environment and notebook checkpoints out of git |

## How to start

You do not push to this repository. You make **your own** repository, as in Tasks 1–3, and submit the URL of its
final commit in Canvas.

1. Download this repository as a ZIP (**Code → Download ZIP**), unzip it, and rename the unzipped folder (GitHub names it
   `task-04-ML-course-main`) to `lastname_ML_task_4`. `README.md` and `data/` must sit directly inside it.
2. Create the virtual environment and install the pinned libraries. You need **Python 3.11, 3.12, 3.13 or 3.14**
   (check with `python3 --version`; on Windows `python --version`):

   ```bash
   cd lastname_ML_task_4
   python3 -m venv .lastname_ML_task_4
   source .lastname_ML_task_4/bin/activate      # Windows PowerShell: .lastname_ML_task_4\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Work in a notebook called `lastname_task4.ipynb` in the root of the folder, next to `data/`.
4. `git init`, commit as you go, push to your own GitHub repository and submit the final commit URL.
   What must be in the repository is in the handout.

## The data

`data/spambase.csv` is derived from the **Spambase** dataset of the UCI Machine Learning Repository, licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
(licence and DOI as stated on the UCI dataset page).

> Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). Spambase [Dataset]. UCI Machine Learning
> Repository. https://doi.org/10.24432/C53G6X

**Changes made to the original files** (`spambase.data`, `spambase.names`):

- The 57 feature names were taken from `spambase.names`; the last column of `spambase.data` is named `spam`.
- Six column names that contained a symbol were renamed: `char_freq_;` → `char_freq_semicolon`,
  `char_freq_(` → `char_freq_paren`, `char_freq_[` → `char_freq_bracket`, `char_freq_!` → `char_freq_exclam`,
  `char_freq_$` → `char_freq_dollar`, `char_freq_#` → `char_freq_hash`. The other 51 names are as in the source.
- Raw rows: 4,601. **391** exact duplicate rows were dropped (first copy kept), leaving 4,210.
- **6** further rows were dropped because their 57 feature values occur with both labels (three pairs of identical
  feature vectors, one copy labelled spam and one legitimate), leaving **4,204** rows: 1,676 spam (39.9 %) and
  2,528 legitimate.
- The rows were shuffled once with the fixed seed 3231 (the raw file lists all spam first).

Nothing else was changed: values are not scaled, clipped or rounded.

This is a teaching dataset built from an old corpus, not evidence about today's spam filters.
