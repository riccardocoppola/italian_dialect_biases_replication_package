import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FILE_2AGENT = "risultati_analisi_completi_2agent.csv"

DIMENSIONS = [
    "Coscienzioso",
    "Mentalita_aperta",
    "Amichevole",
    "Urbano",
    "Calmo",
    "Istruito"
]

DIMENSION_LABELS = {
    "Coscienzioso": "Conscientious",
    "Mentalita_aperta": "Open-minded",
    "Amichevole": "Friendly",
    "Urbano": "Urban",
    "Calmo": "Calm",
    "Istruito": "Educated"
}

LANGUAGES = ["Italiano (Auto)", "Napoletano", "Parmigiano", "Siciliano"]
LANG_LABELS = {
    "Italiano (Auto)": "ITA",
    "Napoletano": "NEA",
    "Parmigiano": "PAR",
    "Siciliano": "SIC"
}

COLOURS = {
    "Italiano (Auto)": "#a6cee3",
    "Napoletano":      "#1f78b4",
    "Parmigiano":      "#b2df8a",
    "Siciliano":       "#33a02c"
}

df2 = pd.read_csv(FILE_2AGENT)

def compute_means(df, suffix):
    results = {}
    for lang in LANGUAGES:
        sub = df[df["Language"] == lang]
        results[lang] = {}
        for dim in DIMENSIONS:
            col = f"{dim}_{suffix}"
            if col in df.columns:
                results[lang][dim] = sub[col].mean()
    return pd.DataFrame(results).T

means2_raw     = compute_means(df2, "Raw")
means2_refined = compute_means(df2, "Refined")

def radar_plot(means_raw, means_corrected, title, filename, label_corrected):
    categories = [DIMENSION_LABELS[d] for d in DIMENSIONS]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             subplot_kw=dict(polar=True))
    fig.suptitle(title, fontsize=22, fontweight="bold", y=1.02)
    plt.subplots_adjust(wspace=0.9)

    for ax, (means, label) in zip(axes, [
        (means_raw, "Raw"),
        (means_corrected, label_corrected)
    ]):
        for lang in LANGUAGES:
            if lang not in means.index:
                continue
            values = means.loc[lang, DIMENSIONS].tolist()
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2,
                    label=LANG_LABELS[lang],
                    color=COLOURS[lang])
            ax.fill(angles, values, alpha=0.1, color=COLOURS[lang])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=18)
        ax.tick_params(axis="x", pad=55)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylim(0, 5)
        ax.set_title(label, fontweight="bold", fontsize=22, pad=60)
        ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.2), fontsize=16)

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura salvata: {Path(filename).resolve()}")

radar_plot(means2_raw, means2_refined,
           "Character Scoring — Raw vs Refined",
           "rq3_radar_2agent.png",
           "Refined")