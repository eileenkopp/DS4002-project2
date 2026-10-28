"""
run_scripts.py - Run the whole project in order to reproduce our results

Purpose : Install the requirements, then run preprocessing, the STL decomposition,
          and the SARIMAX models, stopping if any step fails.

How to run: clone the repo, create and activate a virtual environment
              python3 -m venv .venv
              source .venv/bin/activate
              then run from the main folder: <python SCRIPTS/run_scripts.py>
            To choose where the SARIMAX results are saved:
              python3 SCRIPTS/run_scripts.py --output OUTPUT/sarimax/my_run

Important notes
  - sarimax.py never overwrites an existing run, and our own results are already in
    OUTPUT/sarimax/. So the SARIMAX output folder must be NEW. If you do not pass
    --output, a folder named OUTPUT/sarimax/rerun_<date>_<time> is created automatically.
  - The raw crash file (DATA/Motor_Vehicle_Collisions_-_Crashes_20261016.csv) is too
    large for GitHub. If it is missing, preprocessing is skipped and the processed
    file already in DATA/ is used. If it is present, preprocessing re-creates
    DATA/weekly_injuries_processed.csv and OUTPUT/preprocessing_log.txt.
  - STL results in OUTPUT/stl/ are overwritten on every run.
  - SARIMAX uses a fixed random seed, so results should match ours closely, but tiny
    differences can appear on other computers or package versions.
  - The default SARIMAX run (profile revised, mode full) runs 54 validation fits plus
    3 final fits and can take a while. Use --mode smoke for a quick test run.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---- Locations --------------------------------------------------------------
# This file is in SCRIPTS/, so the project folder is one level up. All scripts are run
# from the project root, because their file paths (DATA/..., OUTPUT/...) start there.
# As described in the main README.md file, all files are run from the project root and only work when run in the project root, including this one.

REQUIREMENTS = Path("requirements.txt")
RAW_DATA =  Path("DATA/Motor_Vehicle_Collisions_-_Crashes_20261016.csv")
PROCESSED_DATA = Path("DATA/weekly_injuries_processed.csv")


# The scripts, in the order they must run
PREPROCESS = "preprocess.py"  # raw crash file -> weekly injury table + cleaning log
STL = "STL.py"                # STL (Loess) decomposition -> OUTPUT/stl/
SARIMAX = "sarimax.py"        # cross-validated SARIMAX models and forecasts -> --output folder


# ---- Options ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run the full project in order.")
parser.add_argument("--output", default=None, help="NEW folder for the SARIMAX results (relative to the project folder). Must not already exist. Default: OUTPUT/sarimax/rerun_<date>_<time>")
parser.add_argument("--profile", choices=["revised", "original"], default="revised", help="SARIMAX model set: revised (default) or the original 3-candidate search")
parser.add_argument("--mode", choices=["smoke", "benchmark", "validate", "full"], default="full", help="SARIMAX mode: smoke = quick test, validate = no forecasts, full = everything")
parser.add_argument("--jobs", type=int, choices=[1, 2, 3], default=3, help="how many SARIMAX validation fits run at the same time (use 1 on a slow computer)")
args = parser.parse_args()

if args.output is None:
    args.output = f"OUTPUT/sarimax/rerun_{datetime.now():%Y%m%d_%H%M%S}"
sarimax_out = args.output


def stop(message):
    """Print an error message and end the program with a failing exit code."""
    print(f"\nERROR: {message}")
    sys.exit(1)


def run_step(script, extra_args=()):
    """Run one script with the same Python that is running this file, from the project folder."""
    print(f"Starting {script}...")
    result = subprocess.run([sys.executable, str(Path("SCRIPTS") / script), *extra_args])
    if result.returncode != 0:
        stop(f"{script} failed. Stopping the sequence.")
    print(f"{script} finished successfully.\n")

# ---- Checks before anything slow starts -------------------------------------
# Fail now, not after preprocessing and STL have already run.
if Path(sarimax_out).exists():
    stop(f"The SARIMAX output folder already exists: {sarimax_out}\nsarimax.py never overwrites a run. Choose a new folder, for example:\npython3 SCRIPTS/run_scripts.py --output OUTPUT/sarimax/my_run")

run_preprocessing = RAW_DATA.exists()
if not run_preprocessing:
    if not PROCESSED_DATA.exists():
        stop(f"Neither the raw data nor the processed data was found.\n")
    print(f"Raw crash file not found, so {PREPROCESS} will be skipped.")
    print(f"Using the existing processed file: {PROCESSED_DATA}\n")


# ---- Install the packages ---------------------------------------------------
result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)])
if result.returncode != 0:
    stop("Failure when installing requirements. Make sure you are in a virtual environment (python3 -m venv .venv, then source .venv/bin/activate).")
print("Requirements installed\n")



# ---- Run the scripts in order -----------------------------------------------
if run_preprocessing:
    run_step(PREPROCESS)

run_step(STL)

run_step(SARIMAX, ["--profile", args.profile, "--mode", args.mode, "--jobs", str(args.jobs), "--output", args.output])




# ---- Check that the SARIMAX run actually completed --------------------------
# sarimax.py writes run.json; its status must be "complete" before the results are used.
run_json = Path(sarimax_out) / "run.json"
if not run_json.exists():
    stop(f"{run_json} was not created, so the SARIMAX results cannot be trusted.")
status = json.loads(run_json.read_text()).get("status")
if status != "complete":
    stop(f"SARIMAX run status is '{status}', not 'complete'. Check {sarimax_out} for details.")

print("All steps finished.")
print(f"STL results: {Path('OUTPUT') / 'stl'}")
print(f"SARIMAX results: {sarimax_out}  (status: {status})")
print("Our original results in OUTPUT/sarimax/ were not changed.")
