import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

FILE_2AGENT = "risultati_analisi_completi_2agent.csv"
RESULTS_2AGENT = "rq3_friedman_2agent.csv"

DIMENSIONS = [
    "Coscienzioso", "Mentalita_aperta", "Amichevole",
    "Urbano", "Calmo", "Istruito"
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
    "Napoletano": "NAP",
    "Parmigiano": "PAR",
    "Siciliano": "SIC"
}
COLOURS = {
    "Italiano (Auto)": "#95a5a6",
    "Napoletano": "#2ecc71",
    "Parmigiano": "#e67e22",
    "Siciliano": "#3498db"
}

COLOR_SIG = "#2c3e50"
COLOR_NOTSIG = "#ecf0f1"


# =========================
# HELPER — formato p-value con esponente
# =========================

def format_pvalue(p):
    if p >= 1.0:
        return "p=1.000"
    if p == 0.0 or p < 1e-10:
        return "p<10⁻¹⁰"
    if p >= 0.001:
        return f"p={p:.4f}"
    else:
        exp = int(np.floor(np.log10(p)))
        base = p / (10 ** exp)
        return f"p={base:.2f}×10⁻{abs(exp)}"


# =========================
# BAR CHART PER DIMENSIONE SIGNIFICATIVA
# =========================

def plot_barchart(df_data, df_results, suffix, condition_label, filename):
    sig = df_results[
        (df_results["condizione"] == suffix) &
        (df_results["significant"] == True)
    ]["dimensione"].tolist()

    if not sig:
        print(f"No significant dimensions for {condition_label}")
        return

    fig, axes = plt.subplots(1, len(sig), figsize=(len(sig) * 3.5, 5), sharey=False)
    if len(sig) == 1:
        axes = [axes]

    fig.suptitle(f"Mean scores per significant dimension — {condition_label}",
                 fontweight="bold", fontsize=13)

    for ax, dim in zip(axes, sig):
        col = f"{dim}_{suffix}"
        means = df_data.groupby("Language")[col].mean().reindex(LANGUAGES)

        x = np.arange(len(LANGUAGES))
        for i, lang in enumerate(LANGUAGES):
            ax.bar(i, means[lang],
                   color=COLOURS[lang],
                   alpha=0.85,
                   label=LANG_LABELS[lang])

        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS[l] for l in LANGUAGES], fontsize=9)
        ax.set_title(DIMENSION_LABELS[dim], fontweight="bold", fontsize=10)
        ax.set_ylabel("Mean score (1-5)")
        ax.set_ylim(0, 5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {Path(filename).resolve()}")


# =========================
# TABELLA SOLO 2 AGENTI (Raw + Refined)
# =========================

def plot_significance_table_2agent(df_results_2, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    table_data = []
    cell_colors = []

    for dim in DIMENSIONS:
        row = [DIMENSION_LABELS[dim]]
        colors = ["white"]

        for cond in ["Raw", "Refined"]:
            sub = df_results_2[(df_results_2["condizione"] == cond) &
                               (df_results_2["dimensione"] == dim)]
            if sub.empty:
                row.append("—")
                colors.append("#f0f0f0")
            else:
                p = sub["p_value_corrected"].values[0]
                sig = sub["significant"].values[0]
                row.append(format_pvalue(p))
                colors.append(COLOR_SIG if sig else COLOR_NOTSIG)

        table_data.append(row)
        cell_colors.append(colors)

    col_labels = ["Dimension", "Raw", "Refined (2-step)"]
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellColours=cell_colors,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    for i, row_data in enumerate(table_data):
        for j in range(len(row_data)):
            if cell_colors[i][j] == COLOR_SIG:
                table[i + 1, j].set_text_props(color="white", fontweight="bold")

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Friedman test significance — Raw vs Refined (2-step pipeline)\n"
        "(dark = significant p<0.05 after Bonferroni correction, light = not significant)",
        fontweight="bold", fontsize=12, pad=20
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {Path(filename).resolve()}")


# =========================
# MAIN
# =========================

df2 = pd.read_csv(FILE_2AGENT)
res2 = pd.read_csv(RESULTS_2AGENT)

# Bar chart
print("=== BAR CHART ===")
plot_barchart(df2, res2, "Raw",
              "Raw",
              "rq3_barchart_raw.png")

plot_barchart(df2, res2, "Refined",
              "Refined (2-step)",
              "rq3_barchart_refined.png")

# Tabella 2 agenti
print("\n=== SIGNIFICANCE TABLE ===")
plot_significance_table_2agent(res2,
                               "rq3_significance_table_2agent.png")