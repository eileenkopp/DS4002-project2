"""
STL.py - Part 1: STL (Loess) decomposition of weekly NYC injury counts

Purpose : Split weekly pedestrian, cyclist, and motorist injuries into trend,
          seasonal, and residual parts to see (1) whether there is a yearly pattern,
          (2) whether the trend shifts around COVID-19 and congestion pricing, and
          (3) whether the three groups have different seasonal patterns.

Run order: run after preprocess.py

Input   : DATA/weekly_injuries_processed.csv (sorted by week, no missing weeks)
          columns: week_ending, pedestrians, cyclists, motorists,
                   holiday_week, covid_period
Output  : OUTPUT/stl/stl_csv/        STL components for each group
          OUTPUT/stl/stl_plots/      decomposition and comparison figures (PNG)
          OUTPUT/stl/info_summary/   strength, event, seasonal, outlier tables (CSV)
          OUTPUT/stl/stl_summary.txt text summary of the results

How to run: from the project root, run  python SCRIPTS/STL.py, since paths are relative it must be run from this directory
Notes: Settings (PERIOD, SEASONAL_WINDOW, ROBUST) are in the Configuration section and rerunning overwrites everything in OUTDIR
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from contextlib import redirect_stdout


# ---------------------- Configuration ---------------------- #
PERIOD = 52                      # weekly data, yearly seasonality
SEASONAL_WINDOW = 13             # odd; larger = smoother, more stable seasonal shape
ROBUST = True                    # down-weights outlier weeks (e.g., COVID shock, blizzards)
CSV_PATH = "DATA/weekly_injuries_processed.csv"
OUTDIR = "OUTPUT/stl"

EVENTS = {
    "COVID-19 (NYC PAUSE)": pd.Timestamp("2020-03-22"),
    "Congestion pricing start": pd.Timestamp("2025-01-05"),
}

# Column in the weekly file -> label used in outputs
GROUPS = {
    "pedestrians": "Pedestrian",
    "cyclists": "Cyclist",
    "motorists": "Motorist",
}
COLORS = {"Pedestrian": "blue", "Cyclist": "green", "Motorist": "red"}



# ----------------------load data ---------------------- #
def load_weekly_counts(csv_path: str) -> pd.DataFrame:
    """Read the weekly file with week_ending as a datetime index; relabel the count columns."""
    df = pd.read_csv(csv_path, parse_dates=["week_ending"], index_col="week_ending")

    # Check only (nothing is repaired): STL's 52-week period assumes exactly one row per week.
    # A skipped week would shift the seasonal cycle, so if there is a missing week just stop the program
    if not (df.index.to_series().diff().dropna() == pd.Timedelta(days=7)).all():
        raise ValueError("week_ending is not strictly 7 days apart; STL needs a gap-free weekly series.")

    return df.rename(columns=GROUPS)

#get the span of the covid period for analysis
def covid_span(df: pd.DataFrame):
    """(start, end) of the covid_period flag, or None if the flag is empty."""
    idx = df.index[df["covid_period"] == 1]
    return (idx.min(), idx.max()) if len(idx) else None





# ---------------------- STL ---------------------- #
def run_stl(series: pd.Series) -> pd.DataFrame:
    y = series.astype(float)
    res = STL(y, period=PERIOD, seasonal=SEASONAL_WINDOW, robust=ROBUST).fit()
    out = pd.DataFrame(
        {"observed": y, "trend": res.trend, "seasonal": res.seasonal, "resid": res.resid}
    )
    return out


def strength(comp: pd.DataFrame) -> dict:
    """Wang/Hyndman strength measures (0 = none, 1 = perfectly deterministic)."""
    var_r = np.var(comp["resid"])
    f_t = max(0.0, 1 - var_r / np.var(comp["trend"] + comp["resid"]))
    f_s = max(0.0, 1 - var_r / np.var(comp["seasonal"] + comp["resid"]))
    return {"trend_strength": f_t, "seasonal_strength": f_s}


def seasonal_profile(comp: pd.DataFrame) -> pd.Series:
    """Average seasonal effect by week of year, as % of the series' mean trend."""
    wk = comp.index.isocalendar().week.clip(upper=52).astype(int)
    prof = comp["seasonal"].groupby(wk.values).mean()
    scale = comp["trend"].mean()
    return prof / scale * 100


def trend_around_events(comp: pd.DataFrame, window: int = 26) -> list:
    """Mean trend in the `window` weeks before vs after each event."""
    rows = []
    for name, date in EVENTS.items():
        before = comp.loc[date - pd.Timedelta(weeks=window): date - pd.Timedelta(days=1), "trend"]
        after = comp.loc[date: date + pd.Timedelta(weeks=window), "trend"]
        if len(before) < 4 or len(after) < 4:
            rows.append({"event": name, "note": "insufficient data in window"})
            continue
        rows.append(
            {
                "event": name,
                "trend_before": before.mean(),
                "trend_after": after.mean(),
                "pct_change": (after.mean() / before.mean() - 1) * 100
            }
        )
    return rows


def flag_outliers(comp: pd.DataFrame, flags: pd.DataFrame, k: float = 3.5) -> pd.DataFrame:
    """Weeks with large residuals (robust z-score via MAD, the median absolute deviation) -
    candidates for exogenous regressors."""
    r = comp["resid"]
    mad = np.median(np.abs(r - np.median(r))) or 1e-9
    # 0.6745 rescales the MAD so the score is comparable to an ordinary bell-curve z-score
    # (for normal data, MAD is about 0.6745 * standard deviation).
    z = 0.6745 * (r - np.median(r)) / mad
    # Keep weeks whose score exceeds the cut-off k (3.5, roughly 3.5 standard deviations).
    out = comp.loc[np.abs(z) > k, ["observed", "trend", "seasonal", "resid"]].assign(robust_z=z)
    return out.join(flags[["holiday_week", "covid_period"]])



# ---------------------- Plotting ---------------------- #


def add_events(ax):
    """Used to mark off occurances of events that are listed in the configuration part"""
    for name, date in EVENTS.items():
        ax.axvline(date, color="black", ls="--", lw=1, alpha=0.7)
    if COVID_SPAN:
        ax.axvspan(*COVID_SPAN, color="gray", alpha=0.15, lw=0)



def plot_decomposition(label: str, comp: pd.DataFrame, path: Path):
    """General structure to plot each of the 3-groups decompistion over time"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    color = COLORS[label]
    for ax, col, title in zip(
        axes,
        ["observed", "trend", "seasonal", "resid"],
        ["Observed", "Trend", f"Seasonal ({PERIOD}-week period)", "Residual"],
    ):
        ax.plot(comp.index, comp[col], color=color, lw=1.2)
        ax.set_ylabel(title)
        add_events(ax)
        ax.grid(alpha=0.25)
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[3].axhline(0, color="gray", lw=0.8)
    handles = [plt.Line2D([], [], color="black", ls="--", label=n) for n in EVENTS]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(f"STL (Loess) decomposition - weekly {label.lower()} injuries", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trends(comps: dict, path: Path):
    """Trend of each group, indexed to its 2019 average (=100) so scales are comparable."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, comp in comps.items():
        base = comp.loc["2019-01-01":"2019-12-31", "trend"].mean()
        ax.plot(comp.index, comp["trend"] / base * 100, label=label, color=COLORS[label], lw=1.8)
    add_events(ax)
    for name, date in EVENTS.items():
        ax.text(date, ax.get_ylim()[1], f" {name}", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_ylabel("Trend index (2019 average = 100)")
    ax.set_title("STL trend by road-user group")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_seasonal_profiles(profiles: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for label in profiles.columns:
        ax.plot(profiles.index, profiles[label], label=label, color=COLORS[label], lw=1.8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Week of year")
    ax.set_ylabel("Seasonal effect (% of mean trend)")
    ax.set_title("Average yearly seasonal pattern by road-user group")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------- Main ---------------------- #
if __name__ == "__main__":

    out = Path(OUTDIR)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stl_csv").mkdir(exist_ok=True)
    (out / "stl_plots").mkdir(exist_ok=True)
    (out / "info_summary").mkdir(exist_ok=True)
    


    #loading the data
    data = load_weekly_counts(CSV_PATH) # load the data
    flags = data[["holiday_week", "covid_period"]]
    weekly = data[list(GROUPS.values())] 
    COVID_SPAN = covid_span(data) #determine covid span
    print(f"Weekly series: {weekly.index.min().date()} to {weekly.index.max().date()} "
          f"({len(weekly)} weeks)")
    if COVID_SPAN:
        print(f"covid_period flag spans {COVID_SPAN[0].date()} to {COVID_SPAN[1].date()} (shaded in plots)")
    print()



    #run the stl analysis on the 3 groups and then plot the results (pedestrians, cyclists, motorists)
    comps, strengths, profiles, event_rows, outliers = {}, [], {}, [], []
    for label in weekly.columns:
        comp = run_stl(weekly[label])
        comps[label] = comp
        comp.to_csv(out / f"stl_csv" / f"stl_components_{label.lower()}.csv")
        plot_decomposition(label, comp, out / f"stl_plots" / f"stl_{label.lower()}.png")

        strengths.append({"group": label, **strength(comp)})
        profiles[label] = seasonal_profile(comp)
        event_rows += [{"group": label, **r} for r in trend_around_events(comp)]
        outliers.append(flag_outliers(comp, flags).assign(group=label))

    #save the STL data into their own dataFrames, then save them in the additional summary info
    strength_df = pd.DataFrame(strengths).set_index("group")
    profile_df = pd.DataFrame(profiles)
    event_df = pd.DataFrame(event_rows)
    outlier_df = pd.concat(outliers)

    strength_df.to_csv(out / f"info_summary" / "strength_summary.csv")
    profile_df.to_csv(out / f"info_summary" / "seasonal_profiles.csv")
    event_df.to_csv(out / f"info_summary" / "trend_event_summary.csv", index=False)
    outlier_df.to_csv(out / f"info_summary" / "residual_outlier_weeks.csv")


    #save the graphs of the seasonal trends and long-term trend changes
    plot_trends(comps, out / f"stl_plots" /"stl_trends_compared.png")
    plot_seasonal_profiles(profile_df, out / f"stl_plots" / "stl_seasonal_profiles.png")


    # ---------------------- Text summary saved into a file ---------------------- #
    pd.set_option("display.float_format", "{:.3f}".format, "display.width", 140)

    #redirect print statement outputs into a summary txt file
    with open(out / "stl_summary.txt", "w") as f, redirect_stdout(f):
        print("1) Is there a yearly pattern? (seasonal strength; >~0.3 is meaningful)")
        print(strength_df, "\n")

        print("2) Trend change around events (26 weeks before vs after)")
        print(event_df.to_string(index=False), "\n")

        print("3) Seasonal pattern by group: peak and trough weeks of year")
        for label in profile_df.columns:
            p = profile_df[label]
            print(f"   {label:<11} peak week {p.idxmax():>2} ({p.max():+.1f}%)   "
                f"trough week {p.idxmin():>2} ({p.min():+.1f}%)")
        print("\n   Correlation of seasonal profiles between groups:")
        print(profile_df.corr(), "\n")

        n_hol = int(outlier_df["holiday_week"].sum()); n_cov = int(outlier_df["covid_period"].sum())
        print(f"Outlier weeks: {len(outlier_df)} total ({n_hol} flagged holiday_week, {n_cov} in covid_period)")

        print("Largest residual weeks:")
        top = outlier_df.sort_values("robust_z", key=np.abs, ascending=False).head(10)
        print(top[["group", "observed", "resid", "robust_z", "holiday_week", "covid_period"]])

    print(f"\nAll outputs saved to: {out.resolve()}")