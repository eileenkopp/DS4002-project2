import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

# ---- Settings ---------------------------------------------------------------
PATH = "DATA/Motor_Vehicle_Collisions_-_Crashes_20261016.csv"  # download from NYC Open Data (see DATA/README.md)

START_DATE = "2016-03-01"   # FORMS electronic reporting began March 2016
END_DATE = "2025-12-31"     # 2026 data is excluded; the test period is the final 12 weeks of 2025

COVID_START = "2020-03-01"  # COVID-19 disruption period (MI2: March 2020 to mid-2021)
COVID_END = "2021-06-30"

OUTPUT_FILE = "DATA/weekly_injuries_processed.csv"
LOG_FILE = "OUTPUT/preprocessing_log.txt"
# -----------------------------------------------------------------------------

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(str(msg))

# ---- Step 1: Load only the columns we need ----------------------------------
# Columns are matched by name so this works on the full download or a trimmed copy.
header = pd.read_csv(PATH, nrows=0).columns
def find(*words):
    for c in header:
        if all(w in c.upper() for w in words):
            return c
    raise ValueError(f"No column containing {words}. Columns found:\n{list(header)}")

date_col = find("CRASH", "DATE")
groups = {
    "pedestrians": find("PEDESTRIAN", "INJURED"),
    "cyclists": find("CYCLIST", "INJURED"),
    "motorists": find("MOTORIST", "INJURED"),
}

# Read everything as text first so invalid entries can be detected, not silently converted
raw = pd.read_csv(PATH, usecols=[date_col, *groups.values()], dtype=str)
raw = raw.rename(columns={date_col: "date", **{v: k for k, v in groups.items()}})

log("=" * 60)
log("PREPROCESSING LOG")
log("=" * 60)
log(f"Source file: {PATH}")
log(f"Columns used: {date_col}, " + ", ".join(groups.values()))
log(f"Records loaded: {len(raw):,}")

# ---- Step 2: Remove missing or invalid dates --------------------------------
# NYC Open Data uses MM/DD/YYYY; a trimmed copy saved by pandas uses YYYY-MM-DD
parsed = pd.to_datetime(raw["date"], format="%m/%d/%Y", errors="coerce")
parsed = parsed.fillna(pd.to_datetime(raw["date"], format="ISO8601", errors="coerce"))
raw["date"] = parsed.dt.normalize()
n_bad_dates = raw["date"].isna().sum()
df = raw.dropna(subset=["date"]).copy()
log(f"\n[Dates]")
log(f"Records with missing or invalid dates removed: {n_bad_dates:,}")

# Restrict to the study period
n_before = len(df)
df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
log(f"Records outside {START_DATE} to {END_DATE} removed: {n_before - len(df):,}")
log(f"Records remaining: {len(df):,}")

# ---- Step 3 and 4: Inspect and clean injury counts --------------------------
log(f"\n[Injury counts]  (bad values set to missing for that group only)")
log(f"{'Group':<12}{'Missing':>10}{'Invalid':>10}{'Negative':>10}{'Total bad':>11}{'% bad':>8}")
for g in groups:
    text = df[g].str.strip()
    missing = text.isna() | (text == "")
    values = pd.to_numeric(text, errors="coerce")
    invalid = ~missing & (values.isna() | (values % 1 != 0))   # non-numeric or fractional
    negative = ~missing & ~invalid & (values < 0)
    bad = missing | invalid | negative

    df[g] = values.where(~bad)        # bad entries become NaN; other groups untouched
    log(f"{g:<12}{missing.sum():>10,}{invalid.sum():>10,}{negative.sum():>10,}"
        f"{bad.sum():>11,}{100 * bad.mean():>7.2f}%")

all_bad = df[list(groups)].isna().all(axis=1).sum()
log(f"Records with no valid injury count in any group: {all_bad:,}")

# ---- Step 5: Aggregate to weekly totals -------------------------------------
# Weeks run Monday to Sunday and are labelled by the Sunday they end on.
# sum() skips missing values, so each group uses only its own valid records.
daily = df.set_index("date")[list(groups)]
weekly = daily.resample("W-SUN").sum(min_count=0)
days_with_data = pd.Series(1, index=daily.index.unique()).resample("W-SUN").sum()
weekly.index.name = "week_ending"

# Drop partial weeks at the edges of the study period (fewer than 7 days covered)
week_start = weekly.index - pd.Timedelta(days=6)
full = (week_start >= pd.Timestamp(START_DATE)) & (weekly.index <= pd.Timestamp(END_DATE))
dropped_edges = (~full).sum()
weekly = weekly[full]
days_with_data = days_with_data[full]

log(f"\n[Weekly aggregation]")
log(f"Partial weeks at the start/end of the period dropped: {dropped_edges}")
log(f"Weeks in final dataset: {len(weekly)} "
    f"({weekly.index.min():%Y-%m-%d} to {weekly.index.max():%Y-%m-%d})")

# Sufficiency check: flag any week where some days have no records at all
gaps = days_with_data[days_with_data < 7]
if len(gaps):
    log(f"WARNING: {len(gaps)} week(s) have days with no records, which may mean missing data:")
    for wk, d in gaps.items():
        log(f"  week ending {wk:%Y-%m-%d}: {d} of 7 days have records")
else:
    log("All weeks have records on all 7 days.")

# ---- Step 6: Indicator variables --------------------------------------------
weekly = weekly.round().astype(int)
start = weekly.index - pd.Timedelta(days=6)

holidays = USFederalHolidayCalendar().holidays(start=START_DATE, end=END_DATE)
weekly["holiday_week"] = [
    int(((holidays >= s) & (holidays <= e)).any()) for s, e in zip(start, weekly.index)
]
# A week counts as COVID if any part of it falls in the COVID period
weekly["covid_period"] = (
    (weekly.index >= pd.Timestamp(COVID_START)) & (start <= pd.Timestamp(COVID_END))
).astype(int)

log(f"\n[Indicators]")
log(f"Holiday weeks (U.S. federal holidays): {weekly['holiday_week'].sum()}")
log(f"COVID-period weeks ({COVID_START} to {COVID_END}): {weekly['covid_period'].sum()}")

# ---- Step 7: Consistency checks ---------------------------------------------
log(f"\n[Consistency checks]")
kept = df[(df["date"] >= weekly.index.min() - pd.Timedelta(days=6))
          & (df["date"] < weekly.index.max() + pd.Timedelta(days=1))]
all_ok = True
for g in groups:
    record_total = int(kept[g].sum())
    weekly_total = int(weekly[g].sum())
    ok = record_total == weekly_total
    all_ok &= ok
    log(f"{g:<12} records: {record_total:>10,}   weekly sum: {weekly_total:>10,}   "
        f"{'OK' if ok else 'MISMATCH'}")

expected_weeks = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-SUN")
no_missing_weeks = len(expected_weeks) == len(weekly)
no_negatives = (weekly[list(groups)] >= 0).all().all()
log(f"No missing weeks in the index: {'OK' if no_missing_weeks else 'FAILED'}")
log(f"No negative weekly totals: {'OK' if no_negatives else 'FAILED'}")
all_ok &= no_missing_weeks and no_negatives
log(f"\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED: review the log above")

# ---- Save -------------------------------------------------------------------
weekly.to_csv(OUTPUT_FILE)
with open(LOG_FILE, "w") as f:
    f.write("\n".join(log_lines))
print(f"\nSaved {OUTPUT_FILE} and {LOG_FILE}")
print(weekly.head())
