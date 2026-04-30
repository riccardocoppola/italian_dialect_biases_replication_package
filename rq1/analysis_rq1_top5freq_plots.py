import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CSV_FILE = "job_assignment_no_bias_correction.csv"

LANGUAGES = ["Italian", "Sicilian", "Parmigiano", "Napoletano"]
LANG_LABELS = {
    "Italian": "ITA",
    "Napoletano": "NAP",
    "Parmigiano": "PAR",
    "Sicilian": "SIC"
    
}
COLOURS = {
    "Italian": "#95a5a6",
    "Napoletano": "#2ecc71",
    "Parmigiano": "#e67e22",
    "Sicilian": "#3498db"
}

df = pd.read_csv(CSV_FILE)
job_cols = [c for c in df.columns if c not in ["profile", "language"]]

for profile in sorted(df["profile"].unique()):
    sub = df[df["profile"] == profile]

    # top 5 job per media su tutte le varietà
    mean_counts = sub[job_cols].mean()
    top5_jobs = mean_counts.sort_values(ascending=False).head(5).index.tolist()

    x = np.arange(5)
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, lang in enumerate(LANGUAGES):
        lang_row = sub[sub["language"] == lang]
        if lang_row.empty:
            continue
        values = [lang_row[j].values[0] if j in lang_row.columns else 0
                  for j in top5_jobs]
        ax.bar(x + offsets[i] * width, values,
               width=width,
               color=COLOURS[lang],
               label=LANG_LABELS[lang],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(top5_jobs, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Frequency (out of 30 runs)")
    ax.set_title(f"Top 5 most assigned jobs — Profile {profile}",
                 fontweight="bold", fontsize=13)
    ax.legend(title="Variety")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    filename = f"rq1_barchart_top5avg_profile{profile}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {Path(filename).resolve()}")