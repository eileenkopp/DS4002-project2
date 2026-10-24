# Data Set Establishment Details

This folder contains the data for our project forecasting weekly traffic injuries in New York City.

| File | Description |
|---|---|
| Raw data (not stored here) | NYPD Motor Vehicle Collisions – Crashes, downloaded from NYC Open Data. See **How to obtain the raw data** below. |
| `weekly_injuries_processed.csv` | Final analysis dataset: weekly injury totals for pedestrians, cyclists, and motorists, with holiday and COVID-19 indicators. Created by `SCRIPTS/preprocess.py`. |

## Data Summary

The Motor Vehicle Collisions – Crashes dataset is maintained on the NYC Open Data portal, with data provided by the New York City Police Department (NYPD) [1]. Each row is one police-reported crash, identified by `COLLISION_ID`. As of October 15, 2026, the dataset has approximately 2.27 million rows and 29 columns, including the crash date and time, location, contributing factors, and the number of people injured or killed by road-user type.

For this project, we used only the crash date and the number of pedestrians, cyclists, and motorists injured. Our preprocessing script removes records with invalid dates, keeps crashes from March 1, 2016 through December 31, 2025, and sums the injuries into Monday-to-Sunday weekly totals. The final dataset has 512 weeks (weeks ending March 13, 2016 through December 28, 2025).

### How to Obtain the Raw data

The raw file (about 567 MB) is too large for GitHub, so it is not stored in this folder. To reproduce our work:

1. Go to the [NYC Open Data Motor Vehicle Collisions – Crashes page](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95).
2. Click **Export** and download the data as a CSV. We downloaded our copy on October 16, 2026.
3. Save it in this `DATA` folder as `Motor_Vehicle_Collisions_-_Crashes_20261016.csv`.
4. From the main repository folder, run `python SCRIPTS/preprocess.py`. This recreates `weekly_injuries_processed.csv`.

Because the NYPD updates records over time, a newer download may produce slightly different counts.

## Provenance

The data comes from the NYPD's motor vehicle collision reporting process. It includes police-reported crashes documented on the New York State MV-104AN collision report, which is required when a crash results in an injury, a death, or at least $1,000 in property damage [1]. In March 2016, the NYPD began recording collisions electronically through its Finest Online Records Management System (FORMS) [1]. We start our data in March 2016 so that this change in reporting does not distort our trends. The records are preliminary and may be amended as information is corrected, so the dataset is not a complete or fixed record of every crash. The NYPD is the data provider, and NYC Open Data hosts and distributes the dataset.

## License

The dataset is publicly available under the NYC Open Data Terms of Use [2]. Under these terms, the data is provided for informational purposes, and the City does not guarantee its completeness, accuracy, or suitability for any particular purpose. Users who reproduce or redistribute the data should identify the original source and describe any changes they made. Our project cites the NYPD and NYC Open Data as the source, and our preprocessing steps are documented in `SCRIPTS/preprocess.py` and `OUTPUT/preprocessing_log.txt`.

## Ethical Statement

Although this dataset is public, traffic collision records concern real people and can include precise crash locations. Our analysis uses only aggregated weekly injury counts and does not identify individuals or highlight individual crashes. We communicate uncertainty in our forecasts, acknowledge possible reporting and data-quality biases, and avoid using our results to make unsupported claims about the safety of particular groups. Our findings are one source of evidence that may inform traffic-safety planning, not a substitute for official NYPD statistics or broader public-health analysis.

## Data Dictionary

### Raw data: columns used

These are the 4 of the 29 raw columns used in our analysis [1].

| Column Name | Description | Data Type | Notes / Uncertainty |
|---|---|---|---|
| CRASH DATE | Date the collision occurred | Floating Timestamp | Records with missing or invalid dates are removed. |
| NUMBER OF PEDESTRIANS INJURED | Number of pedestrians injured | Number | Preliminary and may be revised by the NYPD. Missing, non-numeric, or negative values are treated as missing for this group only. |
| NUMBER OF CYCLIST INJURED | Number of cyclists injured | Number | Same as above. |
| NUMBER OF MOTORIST INJURED | Number of motorists injured | Number | Same as above. |

### Processed data: `weekly_injuries_processed.csv`

| Column Name | Description | Data Type | Notes / Uncertainty |
|---|---|---|---|
| week_ending | The Sunday that ends each Monday-to-Sunday week | Date | Partial weeks at the start and end of the study period are dropped. |
| pedestrians | Total pedestrians injured that week | Integer | Sum of reported injuries, not number of crashes, since one crash can injure several people. Inherits the uncertainty of the raw counts. |
| cyclists | Total cyclists injured that week | Integer | Same as above. |
| motorists | Total motorists injured that week | Integer | Same as above. |
| holiday_week | 1 if the week contains a U.S. federal holiday, otherwise 0 | Integer (0/1) | Based on the federal holiday calendar, not on the crash data. |
| covid_period | 1 if any part of the week falls between March 1, 2020 and June 30, 2021, otherwise 0 | Integer (0/1) | The end date of mid-2021 is an approximation of when travel patterns stabilized. |

## Explanatory Plots

x

**Figure 1.** Weekly pedestrian, cyclist, and motorist injuries, with an 8-week average and the COVID-19 period shaded. All three series show a sharp break in spring 2020. Cyclist injuries returned to their earlier range, while pedestrian and motorist injuries have stayed below pre-pandemic levels since 2022.

x

**Figure 2.** Average weekly injuries by week of the year, before and after COVID-19, with the COVID period excluded. The seasonal shape is stable across periods. Pedestrian injuries peak in late fall and early winter, motorist injuries peak in early summer, and cyclist injuries peak in midsummer.

## References

[1] New York City Police Department, "Motor Vehicle Collisions – Crashes," NYC Open Data. Accessed: Oct. 7, 2026. [Online]. Available: https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95

[2] NYC Open Data, "Terms of Use." Accessed: Oct. 15, 2026. [Online]. Available: https://data.cityofnewyork.us/stories/s/Terms-of-Use/k9k7-3cje
