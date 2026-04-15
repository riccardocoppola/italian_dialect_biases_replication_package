import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

FILE_2AGENT = "risultati_analisi_completi_2agent.csv"
FILE_3AGENT = "risultati_finali_multiagente_3agent.csv"

DIMENSIONS = [
    "Coscienzioso",
    "Mentalita_aperta",
    "Amichevole",
    "Urbano",
    "Calmo",
    "Istruito"
]

LANGUAGES = ["Napoletano", "Italiano (Auto)", "Parmigiano", "Siciliano"]

# =========================
# LOAD DATA
# =========================

df2 = pd.read_csv(FILE_2AGENT)
df3 = pd.read_csv(FILE_3AGENT)

print("=== 2 AGENTI ===")
print(f"Righe: {len(df2)}, Colonne: {list(df2.columns)}")
print(f"Lingue: {df2['Language'].unique()}")

print("\n=== 3 AGENTI ===")
print(f"Righe: {len(df3)}, Colonne: {list(df3.columns)}")
print(f"Lingue: {df3['Language'].unique()}")

# =========================
# MEDIE PER LINGUA E DIMENSIONE
# =========================

def compute_means(df, suffix):
    results = {}
    for lang in df["Language"].unique():
        sub = df[df["Language"] == lang]
        results[lang] = {}
        for dim in DIMENSIONS:
            col = f"{dim}_{suffix}"
            if col in df.columns:
                results[lang][dim] = sub[col].mean()
    return pd.DataFrame(results).T

means2_raw     = compute_means(df2, "Raw")
means2_refined = compute_means(df2, "Refined")
means3_raw     = compute_means(df3, "Raw")
means3_final   = compute_means(df3, "Final")

print("\n=== MEDIE 2 AGENTI - RAW ===")
print(means2_raw)
print("\n=== MEDIE 2 AGENTI - REFINED ===")
print(means2_refined)
print("\n=== MEDIE 3 AGENTI - RAW ===")
print(means3_raw)
print("\n=== MEDIE 3 AGENTI - FINAL ===")
print(means3_final)

# =========================
# RADAR PLOT
# =========================

def radar_plot(means_raw, means_corrected, title, filename, label_corrected):
    categories = DIMENSIONS
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors = {
        "Napoletano": "#3498db",
        "Italiano (Auto)": "#95a5a6",
        "Parmigiano": "#e67e22",
        "Siciliano": "#2ecc71"
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7),
                             subplot_kw=dict(polar=True))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    for ax, (means, label) in zip(axes, [
        (means_raw, "Raw (Agent 1)"),
        (means_corrected, label_corrected)
    ]):
        for lang in means.index:
            values = means.loc[lang, DIMENSIONS].tolist()
            values += values[:1]
            color = colors.get(lang, "black")
            ax.plot(angles, values, "o-", linewidth=2,
                    label=lang, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 5)
        ax.set_title(label, fontweight="bold", pad=15)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figura salvata: {Path(filename).resolve()}")

radar_plot(means2_raw, means2_refined,
           "2-Agent Pipeline: Raw vs Refined",
           "rq3_radar_2agent.png",
           "Refined (Agent 2)")

radar_plot(means3_raw, means3_final,
           "3-Agent Pipeline: Raw vs Final",
           "rq3_radar_3agent.png",
           "Final (Agent 3)")