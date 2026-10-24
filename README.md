# DS4002-project2


## Content Summary
This repository contains our data information, scripts, and results/output for our project of time-series data, weekly traffic injuries in New York City among 3 road user groups (**pedestrians, cyclists, and motorists**). We start with the NYPD's crash records on NYC Open Data (March 2016 to December 2025), process/clean the data to combine them into weekly injury counts for each group. Then do 3 things with each of the those counts, first we use STL with Loess to analyze the long-term trend, seasonal pattern, and residuals of the injuries, then second we create a SARIMAX forecasting model to see if we can forecast crash amounts better than a seasonal naive baseline, finally saving all of this info into the output folder and creating various plots to help interpret the results. 

## Section 1: Software and platform section 

- We used Python 3.13 as our programming language, which is the environment the SARIMAX code was verified in. Other python versions may not work so be sure to set up a virtual env with this version of python

Add-on packages that need to be installed (the exact versions are in requirements.txt):
- pandas
- numpy
- matplotlib
- statsmodels (provides the STL and SARIMAX implementations; it also installs scipy)

The platform we used:
- Mac

## Section 2: Documentation Map. 

An outline illustrating the hierarchy of folders and subfolders contained in your Project Folder, and listing the files stored in each folder or subfolder. 

- DATA
  - Motor_Vehicle_Collisions_-_Crashes_20261016.csv
      - **This file is too big to upload to github. To access, use this link to view where we got the data from the in the NYPD database (steps below contain how to get the data along with in the DATA/README.md file**)
  - weekly_injuries_processed.csv
  - README.md
- OUTPUT
  - fig1_weekly_series.png
  - fig2_seasonal_profile.png
  - preprocessing_log.txt
  - weekly_injuries_processed.csv
  - sarimax
    - benchmark
        - benchmark_candidate_0.csv
        - benchmark_candidate_1.csv
        - benchmark_candidate_2.csv
        - candidate_scores.csv
        - diagnostics.json
        - fold_scores.csv
        - run.json
    - full
        - best_orders.json
        - candidate_scores.csv
        - diagnostics.json
        - fold_scores.csv
        - forecasts.csv
        - run.json
    - revised
        - RESULTS.md
        - full
            - best_orders.json
            - candidate_scores.csv
            - comparison.csv
            - diagnostics.json
            - fold_scores.csv
            - forecasts.csv
            - run.json
            - validation_forecasts.csv
  - stl
      - stl_summary.txt
      - info_summary
          - residual_outlier_weeks.csv
          - seasonal_profiles.csv
          - strength_summary.csv
          - trend_event_summary.csv
      - stl_csv
          - stl_components_cyclist.csv
          - stl_components_motorist.csv
          - stl_components_pedestrian.csv
      - stl_plots
          - stl_cyclist.png
          - stl_motorist.png
          - stl_pedestrian.png
          - stl_seasonal_profiles.png
          - stl_trends_compared.png
- SCRIPTS
    - EDA.py
    - SARIMAX.md
    - STL.py
    - preprocess.py
    - run_scripts.py
    - sarimax.py
- LICENSE.md
- README.md
- requirements.txt
       
## Section 3: Instructions for reproducing results.  

**Important Notes before starting**: All commands run below are run from the **top-level folder** (folder that contains `DATA`, `OUTPUT`, `SCRIPTS`), not from inside `SCRIPTS`. If not in top-level folder will cause relative path errors when running scripts

1. Clone the repo: clone the repository to your local PC and open it in your IDE
2. Create a virtual environment that runs with version Python 3.13 <python3.13 -m venv venv>
3. Activate the virtual environment <source venv/bin/activate>, verify the version is right with <python --version> (if something goes wrong with steps 2-3, google the issue to ensure these 2 steps are right)
4. Install the required packages: <pip install -r requirements.txt>
5. Download the Data from the internet, go to "https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95/about_data" and export the data as a csv
6. Move the downloaded data into DATA folder under the name <Motor_Vehicle_Collisions_-_Crashes_20261016.csv>
   - **Note**: If you don't wish to get the data online, you can do the following steps by just using our already cleaned data file: `DATA/weekly_injuries_processed.csv` and can skip directly to automatically running everything (see step 7) or directly to the stl analysis (step 9)
7. **Choose how to run the scripts**: There are 2 options
   - **Option 1 (automatic)**: run <python SCRIPTS/run_scripts.py> + <arguments_you_choose>. See the file `SCRIPTS/run_scripts.py` for the additional arguments you can run with the program. This option automatically goes through the scripts and runs them for you. See **STEP 10** for the options you have. Example run <python SCRIPTS/run_scripts.py --jobs 2 --output OUTPUT/sarimax/my_run>
   - **Option 2 (manual)**: run the scripts yourself, in order, as described in the following steps.
8. Run `preprocess.py` (**option 2 only from beyond here**)
   - **SKIP IF YOU ARE USING THE ALREADY PROCESSED DATA**
   - This cleans the data from 2016 to 2025 and aggregates the weekly injury counts of the 3 different groups
   - Saves the cleaned data as `DATA/weekly_injuries_processed.csv` (the file the other scripts read) and a cleaning log as `OUTPUT/preprocessing_log.txt`. A copy of our cleaned data is also kept in `OUTPUT` for reference.
     - (Optional, requires the raw CSV) Run <python SCRIPTS/EDA.py> to recreate the two exploratory plots shown in `DATA/README.md`, saved as `OUTPUT/fig1_weekly_series.png` and `OUTPUT/fig2_seasonal_profile.png`
9. Run `STL.py`
   - This does the STL with loess analysis on the cleaned data to determine trend over time, seasonal patterns, and residuals (weird cases) in the cleaned data
   - Creates `OUTPUT/stl` folder which contains various plots, a text summary of the results, csv's of the STL results (trend, seasonal, residuals), and additional info about the over all results in csv format
10. Run `sarimax.py`
    - **Script never overwrites an existing run**. The repo already has our SARIMAX results, so when you try to reproduce them you must give each run a new folder with the <--output> command. Compare your new folder with ours afterward
    - **Default version run is the revised profile**. Add <--profile original> to run the original 3 candidate search instead
    - **Check `SCRIPTS/SARIMAX.md` to view all available commands**. There are a many ways that you can run the model that are too extensive to list in this summary portion, if you want to see how to run it in more detail check the file, otherwise just run it following the details provided above
    - Have the following arguments you can add onto the <python SCRIPTS/sarimax.py> command:
     1. <--profile __your_choice__>: "revised", "original" (SARIMAX model set: revised (default) or the original 3-candidate search)
     2. <--mode __your_choice__>: "smoke", "benchmark", "validate", "full" (SARIMAX mode: smoke (default) = quick test, validate = no forecasts, full = everything; **use full to reproduce our results**)
     3. <--jobs __your_choice__>: 1, 2, 3 (how many SARIMAX validation fits run at the same time (use 1 on a slow computer), 1 is default)
     4. <--output ___output_dir__>: your output file name and location, remember **must be a new location**
    - Example command to reproduce our results: <python SCRIPTS/sarimax.py --profile revised --mode full --jobs 3 --output OUTPUT/my_run>
11. Interpret the results from the figures produced in the `OUTPUT` folder. STL results will be found under the folder `OUTPUT/stl`, SARIMAX results will be found under `OUTPUT/sarimax`. Preprocessing logs and cleaned csv will also be in `OUTPUT` if you are curious about those results as well.

## References
1. [1] New York City Police Department, "Motor Vehicle Collisions – Crashes," NYC Open Data. Accessed: Oct. 7, 2026. [Online]. Available: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95
