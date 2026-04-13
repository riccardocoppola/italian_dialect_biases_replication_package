
import json
import csv
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# =========================
# CONFIG
# =========================

INPUT_FILE = "job_assignment_no_bias_correction.json"
CSV_FILE = "job_assignment_no_bias_correction.csv"
PROFILE_RESULTS_FILE = "profile_statistical_results.csv"
GLOBAL_DIFFS_FILE = "global_job_differences_vs_italian.csv"
FIGURE_FILE = "job_distribution_differences_vs_italian.png"

LANGUAGES = ["Italian", "Sicilian", "Parmigiano", "Napoletano"]

# Use these labels in the plot
PLOT_LABELS = {
    "Italian": "ITA",
    "Sicilian": "SIC",
    "Parmigiano": "EML",
    "Napoletano": "NAP",
}

# Optional normalisation of obviously inconsistent labels
NORMALISE_JOBS = False


# =========================
# HELPERS
# =========================

def parse_jobs(job_string: str) -> list[str]:
    if not job_string:
        return []
    return [job.strip() for job in job_string.split(",") if job.strip()]


def normalise_job_name(job: str) -> str:
    """
    Optional light normalisation.
    Keep this conservative unless you explicitly want aggressive merging.
    """
    j = job.strip().lower()

    mapping = {
        "dj": "dj",
        "commico": "comico",
        "musician": "musicista",
        "photografo": "fotografo",
        "scritto-re": "scrittore",
        "sarto": "sarto",
        "modella": "modella",
        "street artist": "street artist",
        "club manager": "club manager",
    }

    return mapping.get(j, j)


def cramers_v(chi2: float, n: float, r: int, k: int) -> float:
    denom = n * min(r - 1, k - 1)
    if denom == 0:
        return np.nan
    return np.sqrt(chi2 / denom)


def ensure_language_order(df_sub: pd.DataFrame, languages: list[str]) -> pd.DataFrame:
    return df_sub.set_index("language").reindex(languages)


# =========================
# STEP 1 — JSON -> CSV
# =========================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

counts = defaultdict(Counter)
all_jobs = set()
profile_ids = set()

for record in data:
    profile_id = record["id"]
    profile_ids.add(profile_id)

    for lang in LANGUAGES:
        key = f"jobs_{lang}"
        jobs = parse_jobs(record.get(key, ""))

        for job in jobs:
            if NORMALISE_JOBS:
                job = normalise_job_name(job)
            counts[(profile_id, lang)][job] += 1
            all_jobs.add(job)

job_list = sorted(all_jobs, key=str.casefold)

with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["profile", "language", *job_list])

    for profile_id in sorted(profile_ids):
        for lang in LANGUAGES:
            row_counter = counts.get((profile_id, lang), Counter())
            row = [profile_id, lang] + [row_counter.get(job, 0) for job in job_list]
            writer.writerow(row)

print(f"CSV created: {Path(CSV_FILE).resolve()}")


# =========================
# STEP 2 — LOAD CSV
# =========================

df = pd.read_csv(CSV_FILE)
job_cols = [c for c in df.columns if c not in ["profile", "language"]]


# =========================
# STEP 3 — PER-PROFILE STATISTICAL ANALYSIS
# =========================

profile_results = []

print("\n=== PER-PROFILE RESULTS ===")

for profile in sorted(df["profile"].unique()):
    sub = df[df["profile"] == profile].copy()
    sub = ensure_language_order(sub, LANGUAGES)

    # keep only jobs that appear at least once in this profile
    profile_job_cols = [c for c in job_cols if sub[c].sum() > 0]

    if not profile_job_cols:
        print(f"Profile {profile}: skipped (no counts)")
        continue

    table = sub[profile_job_cols].fillna(0).to_numpy(dtype=float)

    if table.sum() == 0:
        print(f"Profile {profile}: skipped (all zeros)")
        continue

    if table.shape[1] < 2:
        print(f"Profile {profile}: skipped (fewer than 2 job categories)")
        continue

    chi2, p, dof, expected = chi2_contingency(table)

    n = table.sum()
    r, k = table.shape
    v = cramers_v(chi2, n, r, k)

    profile_results.append({
        "profile": profile,
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": v,
        "n_jobs_used": len(profile_job_cols),
        "total_assignments": int(n),
    })

    print(
        f"Profile {profile}: "
        f"p={p:.3e}, chi2={chi2:.2f}, dof={dof}, "
        f"Cramér's V={v:.3f}, jobs={len(profile_job_cols)}"
    )

profile_results_df = pd.DataFrame(profile_results).sort_values("profile")
profile_results_df.to_csv(PROFILE_RESULTS_FILE, index=False, encoding="utf-8")

print(f"\nPer-profile results saved to: {Path(PROFILE_RESULTS_FILE).resolve()}")


# =========================
# STEP 4 — GLOBAL ANALYSIS
# =========================

print("\n=== GLOBAL RESULT ===")

global_counts = df.groupby("language")[job_cols].sum().reindex(LANGUAGES)
global_job_cols = [c for c in job_cols if global_counts[c].sum() > 0]
global_table = global_counts[global_job_cols].to_numpy(dtype=float)

chi2, p, dof, expected = chi2_contingency(global_table)
n = global_table.sum()
r, k = global_table.shape
v = cramers_v(chi2, n, r, k)

print(
    f"Global: p={p:.3e}, chi2={chi2:.2f}, dof={dof}, "
    f"Cramér's V={v:.3f}, jobs={len(global_job_cols)}"
)


# =========================
# STEP 5 — JOB DIFFERENCES VS ITALIAN
# =========================

# convert counts to within-language proportions
# =========================
# STEP 5 — ABSOLUTE DIFFERENCES VS ITALIAN
# =========================

global_counts = df.groupby("language")[job_cols].sum().reindex(LANGUAGES)

diff_sic = global_counts.loc["Sicilian"] - global_counts.loc["Italian"]
diff_par = global_counts.loc["Parmigiano"] - global_counts.loc["Italian"]
diff_nap = global_counts.loc["Napoletano"] - global_counts.loc["Italian"]

diffs_df = pd.DataFrame({
    "ITA": global_counts.loc["Italian"],
    "SIC": global_counts.loc["Sicilian"],
    "EML": global_counts.loc["Parmigiano"],
    "NAP": global_counts.loc["Napoletano"],
    "diff_SIC_vs_ITA": diff_sic,
    "diff_EML_vs_ITA": diff_par,
    "diff_NAP_vs_ITA": diff_nap,
})

# ranking: biggest absolute count difference
diffs_df["max_abs_diff"] = diffs_df[
    ["diff_SIC_vs_ITA", "diff_EML_vs_ITA", "diff_NAP_vs_ITA"]
].abs().max(axis=1)

diffs_df = diffs_df.sort_values("max_abs_diff", ascending=False)


# =========================
# STEP 6 — VISUALISATION
# =========================

# choose top jobs to display
TOP_N = 30
plot_df = diffs_df.head(TOP_N).copy()

# for the grey Italian reference, plot Italian proportions on the same x-axis scale
# this is optional, but keeps the feel of your previous figure
# if you want pure difference-only panels, remove the Italian scatter points
ita_ref = plot_df["ITA"]

# prepare data per panel
panels = [
    ("ITA vs SIC", plot_df["diff_SIC_vs_ITA"], "SIC"),
    ("ITA vs EML", plot_df["diff_EML_vs_ITA"], "EML"),
    ("ITA vs NAP", plot_df["diff_NAP_vs_ITA"], "NAP"),
]

# reverse for top-at-top display
job_names = list(plot_df.index)[::-1]
y = np.arange(len(job_names))

fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
fig.suptitle("Occupational Distribution Differences: Dialects vs. Standard Italian",
             fontsize=18, fontweight="bold", y=0.97)

dialect_colours = {
    "ITA": "#95a5a6",
    "SIC": "#3498db",
    "EML": "#e67e22",
    "NAP": "#2ecc71",
}

for ax, (title, series, dialect_code) in zip(axes, panels):
    series = series.reindex(job_names)
    ita_vals = ita_ref.reindex(job_names)

    # horizontal segments from 0 to difference
    ax.hlines(y, 0, series.values, color=dialect_colours[dialect_code], lw=1.5, alpha=0.8)

    # dialect points
    ax.scatter(series.values, y, s=40, color=dialect_colours[dialect_code], zorder=3, label=dialect_code)

    # optional Italian reference points
    #ax.scatter(np.zeros_like(y), y, s=28, color=dialect_colours["ITA"], zorder=3, alpha=0.9, label="ITA")

    ax.axvline(0, color="black", lw=1)
    ax.set_title(title, fontweight="bold", pad=12)
    ax.set_xlabel("Δ proportion vs Italian")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

axes[0].set_yticks(y)
axes[0].set_yticklabels(job_names, fontsize=10)

# custom legend
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=dialect_colours["ITA"], markersize=8, label="ITA"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=dialect_colours["SIC"], markersize=8, label="SIC"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=dialect_colours["EML"], markersize=8, label="EML"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=dialect_colours["NAP"], markersize=8, label="NAP"),
]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.945))

plt.tight_layout(rect=[0.16, 0.06, 1, 0.95])
plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"Figure saved to: {Path(FIGURE_FILE).resolve()}")
