import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

BASELINE_FILE = "job_assignment_no_bias_correction_with_deltas.csv"

LANGUAGES = ["Italian", "Napoletano", "Parmigiano", "Sicilian"]
LANG_LABELS = {
    "Italian": "ITA",
    "Napoletano": "NEA",
    "Parmigiano": "PAR",
    "Sicilian": "SIC",
}
COLOURS = {
    "Italian":    "#a6cee3",
    "Napoletano": "#1f78b4",
    "Parmigiano": "#b2df8a",
    "Sicilian":   "#33a02c"
}

DIALECTS = ["Napoletano", "Parmigiano", "Sicilian"]
DELTA_COLOURS = {
    "delta_NAP": "#1f78b4",
    "delta_PAR": "#b2df8a",
    "delta_SIC": "#33a02c"
}
DELTA_LABELS = {
    "delta_NAP": "NEA-ITA",
    "delta_PAR": "PAR-ITA",
    "delta_SIC": "SIC-ITA"
}
TOP_N = 5


# =========================
# HELPER
# =========================

def get_profiles(df):
    return sorted([p for p in df["profile"].unique() if str(p).isdigit()], key=int)


def draw_barchart(ax, sub, job_list):
    x = np.arange(len(job_list))
    width = 0.2

    for i, lang in enumerate(LANGUAGES):
        values = sub.loc[lang, job_list].values
        ax.bar(
            x + i * width,
            values,
            width,
            label=LANG_LABELS[lang],
            color=COLOURS[lang],
            alpha=0.85
        )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(job_list, rotation=30, ha="right", fontsize=18)
    ax.set_ylabel("Normalised frequency (0-1)", fontsize=22)
    ax.tick_params(axis="y", labelsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=16)


# =========================
# OPZIONE B — top 5 job con maggiore differenza
# =========================

def plot_bar_top5_diff(df):
    job_cols = [c for c in df.columns if c not in ["profile", "language"]]

    for profile in get_profiles(df):
        sub = df[(df["profile"] == profile) & (df["language"].isin(LANGUAGES))]
        sub = sub.set_index("language").reindex(LANGUAGES)

        ita = sub.loc["Italian", job_cols]
        max_diff = pd.Series(0.0, index=job_cols)
        for dialect in DIALECTS:
            diff = (sub.loc[dialect, job_cols] - ita).abs()
            max_diff = max_diff.combine(diff, max)

        top_jobs = max_diff.sort_values(ascending=False).head(TOP_N).index.tolist()

        if not top_jobs:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(
            f"Top {TOP_N} jobs by difference — Profile {profile}",
            fontweight="bold",
            fontsize=22
        )

        draw_barchart(ax, sub, top_jobs)

        plt.tight_layout()
        filename = f"barchart_top5diff_profile{profile}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Salvato: {Path(filename).resolve()}")


# =========================
# OPZIONE C — tutti i job assegnati
# =========================

def plot_bar_all_jobs(df):
    job_cols = [c for c in df.columns if c not in ["profile", "language"]]

    for profile in get_profiles(df):
        sub = df[(df["profile"] == profile) & (df["language"].isin(LANGUAGES))]
        sub = sub.set_index("language").reindex(LANGUAGES)

        active_jobs = [j for j in job_cols if sub[j].sum() > 0]

        if not active_jobs:
            continue

        fig, ax = plt.subplots(figsize=(max(12, len(active_jobs) * 0.9), 6))
        fig.suptitle(
            f"All assigned jobs — Profile {profile}",
            fontweight="bold",
            fontsize=22
        )

        draw_barchart(ax, sub, active_jobs)

        plt.tight_layout()
        filename = f"barchart_all_profile{profile}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Salvato: {Path(filename).resolve()}")


# =========================
# BOXPLOT PER PROFILO
# =========================

def plot_boxplot_per_profile(df):
    job_cols = [c for c in df.columns if c not in ["profile", "language"]]

    for profile in get_profiles(df):
        sub_delta = df[
            (df["profile"] == profile) &
            (df["language"].isin(["delta_NAP", "delta_PAR", "delta_SIC"]))
        ]

        if sub_delta.empty:
            continue

        data = []
        box_labels = []
        colors = []

        for delta_label in ["delta_NAP", "delta_PAR", "delta_SIC"]:
            row = sub_delta[sub_delta["language"] == delta_label]
            if row.empty:
                continue
            values = row[job_cols].values.flatten()
            values = values[(~np.isnan(values)) & (values != 0)]
            data.append(values)
            box_labels.append(DELTA_LABELS[delta_label])
            colors.append(DELTA_COLOURS[delta_label])

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(data, patch_artist=True, tick_labels=box_labels)

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.axhline(0, color="black", lw=1, linestyle="--")
        ax.set_title(
            f"Normalised delta per job — Profile {profile}",
            fontweight="bold",
            fontsize=22
        )
        ax.set_xlabel("Dialect vs Italian comparison", fontsize=22)
        ax.set_ylabel("Normalised delta (dialect - ITA)", fontsize=22)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        filename = f"boxplot_profile{profile}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Salvato: {Path(filename).resolve()}")


# =========================
# MAIN
# =========================

df = pd.read_csv(BASELINE_FILE)
plot_bar_top5_diff(df)
plot_bar_all_jobs(df)
plot_boxplot_per_profile(df)