"""
eda.py - Exploratory data analysis: weekly injuries before, during, and after COVID-19

Purpose : Answer our EDA question from MI2: how did weekly injuries to pedestrians,
          cyclists, and motorists change before, during, and after the COVID-19
          pandemic, in both overall level and seasonal pattern? Produces the two
          explanatory plots shown in DATA/README.md.

Run order: independent of the other scripts (reads the raw data directly). Can be
           run any time after the raw CSV is downloaded into DATA/.

Input   : DATA/Motor_Vehicle_Collisions_-_Crashes_20261016.csv (raw data, ~567 MB,
          not in repo; download instructions in DATA/README.md)
Output  : OUTPUT/fig1_weekly_series.png      weekly injuries over time, COVID period shaded
          OUTPUT/fig2_seasonal_profile.png   average week-of-year pattern, before vs after COVID
          OUTPUT/eda_period_summary.csv      average weekly injuries in each period

How to run: from the project root, run  python SCRIPTS/eda.py, since paths are relative it must be run from this directory
Notes: Uses the full date range of the raw file (2012 onward), not the 2016-2025 study
       period used for modeling, so the plots show the longer history. Period cutoffs
       (PRE_END, POST_START) are in the Settings section.
"""

import pandas as pd                 # data loading and weekly aggregation
import matplotlib.pyplot as plt     # plotting

# ---- Settings ---------------------------------------------------------------
PATH = "DATA/Motor_Vehicle_Collisions_-_Crashes_20261016.csv"  # raw crash data
PRE_END = "2020-03-01"     # "before COVID" ends the day before this date
POST_START = "2021-07-01"  # "after COVID" starts on this date (between = COVID period)

FIG1 = "OUTPUT/fig1_weekly_series.png"
FIG2 = "OUTPUT/fig2_seasonal_profile.png"
SUMMARY = "OUTPUT/eda_period_summary.csv"
# -----------------------------------------------------------------------------

# ---- Load data --------------------------------------------------------------
# Find the needed columns by name, so the script works even if column order changes
header = pd.read_csv(PATH, nrows=0).columns
def find(*words):
    """Return the first column whose name contains all of the given words."""
    for c in header:
        if all(w in c.upper() for w in words):
            return c
    raise ValueError(f"No column containing {words}. Columns are:\n{list(header)}")

date_col = find("CRASH", "DATE")
groups = {
    "Pedestrians": find("PEDESTRIAN", "INJURED"),
    "Cyclists": find("CYCLIST", "INJURED"),
    "Motorists": find("MOTORIST", "INJURED"),
}
print("Using columns:", {"date": date_col, **groups})

# Read only the date and the three injury columns to save memory
df = pd.read_csv(PATH, usecols=[date_col, *groups.values()], low_memory=False)

# Convert dates; unreadable dates become missing and are dropped
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
print(f"Rows loaded: {len(df):,}  |  unreadable dates dropped: {df[date_col].isna().sum():,}")
df = df.dropna(subset=[date_col]).rename(columns={v: k for k, v in groups.items()})

# Injury counts should be numbers; anything else counts as 0 for this exploratory view
for g in groups:
    df[g] = pd.to_numeric(df[g], errors="coerce").fillna(0)

# ---- Weekly totals ----------------------------------------------------------
# Sum injuries into weeks ending Sunday, then drop the first and last weeks,
# which are usually partial and would look artificially low
weekly = df.set_index(date_col)[list(groups)].resample("W-SUN").sum().iloc[1:-1]
weekly.index.name = "week_ending"

# Label each week as before, during, or after COVID
def period(d):
    if d < pd.Timestamp(PRE_END):
        return "Before COVID"
    if d < pd.Timestamp(POST_START):
        return "COVID period"
    return "After COVID"
weekly["period"] = weekly.index.map(period)

# ---- Summary table ----------------------------------------------------------
# Average weekly injuries in each period, plus the percent change before -> after
order = ["Before COVID", "COVID period", "After COVID"]
summary = weekly.groupby("period")[list(groups)].mean().reindex(order).round(1)
summary.loc["% change (before -> after)"] = (
    (summary.loc["After COVID"] / summary.loc["Before COVID"] - 1) * 100).round(1)
summary.to_csv(SUMMARY)
print(f"Data covers {weekly.index.min():%Y-%m-%d} to {weekly.index.max():%Y-%m-%d} ({len(weekly)} weeks)")
print("\nAverage weekly injuries:")
print(summary.to_string())

colors = {"Pedestrians": "#2a6fdb", "Cyclists": "#e08a1e", "Motorists": "#4a9d5b"}

# ---- Figure 1: weekly series with COVID shaded ------------------------------
# One panel per group: thin line = weekly counts, thick line = 8-week rolling
# average to show the trend, grey band = COVID period
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for ax, g in zip(axes, groups):
    ax.plot(weekly.index, weekly[g], color=colors[g], lw=0.8, alpha=0.5)
    ax.plot(weekly.index, weekly[g].rolling(8, center=True).mean(),
            color=colors[g], lw=2, label="8-week average")
    ax.axvspan(pd.Timestamp(PRE_END), pd.Timestamp(POST_START),
               color="grey", alpha=0.15, label="COVID period")
    ax.set_ylabel("Injured / week")
    ax.set_title(g, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, loc="upper right")
fig.suptitle("Weekly injuries in NYC motor vehicle crashes", fontsize=13)
fig.tight_layout()
fig.savefig(FIG1, dpi=150)
plt.close(fig)

# ---- Figure 2: seasonal profile, before vs after ----------------------------
# Average injuries for each week of the year (1-52), computed separately for the
# before and after periods, so differences in seasonal shape are visible.
# Week 53 (in some years) is merged into week 52.
weekly["week_of_year"] = weekly.index.isocalendar().week.clip(upper=52).astype(int)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, g in zip(axes, groups):
    for p, style in [("Before COVID", "--"), ("After COVID", "-")]:
        prof = weekly[weekly["period"] == p].groupby("week_of_year")[g].mean()
        ax.plot(prof.index, prof.values, style, color=colors[g], lw=2, label=p)
    ax.set_title(g, loc="left", fontsize=11)
    ax.set_xlabel("Week of year")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Average injured / week")
axes[0].legend(frameon=False)
fig.suptitle("Seasonal pattern before vs. after COVID (COVID period excluded)", fontsize=13)
fig.tight_layout()
fig.savefig(FIG2, dpi=150)
plt.close(fig)

print(f"\nSaved {FIG1}, {FIG2}, and {SUMMARY}")
