import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

FILE_2AGENT = "risultati_analisi_completi_2agent.csv"
FILE_3AGENT = "risultati_finali_multiagente_3agent.csv"
RESULTS_2AGENT = "rq3_friedman_2agent.csv"
RESULTS_3AGENT = "rq3_friedman_3agent.csv"

DIMENSIONS = [
    "Coscienzioso", "Mentalita_aperta", "Amichevole",
    "Urbano", "Calmo", "Istruito"
]

LANGUAGES = ["Napoletano", "Italiano (Auto)", "Parmigiano", "Siciliano"]
LANG_LABELS = {
    "Napoletano": "NAP",
    "Italiano (Auto)": "ITA",
    "Parmigiano": "EML",
    "Siciliano": "SIC"
}
COLOURS = {
    "Napoletano": "#2ecc71",
    "Italiano (Auto)": "#95a5a6",
    "Parmigiano": "#e67e22",
    "Siciliano": "#3498db"
}

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
# OPZIONE B — BAR CHART PER DIMENSIONE SIGNIFICATIVA
# =========================

def plot_barchart(df_data, df_results, suffix, condition_label, filename):
    sig = df_results[
        (df_results["condizione"] == suffix) &
        (df_results["significant"] == True)
    ]["dimensione"].tolist()

    if not sig:
        print(f"Nessuna dimensione significativa per {condition_label}")
        return

    fig, axes = plt.subplots(1, len(sig), figsize=(len(sig) * 3.5, 5), sharey=False)
    if len(sig) == 1:
        axes = [axes]

    fig.suptitle(f"Medie per dimensione significativa — {condition_label}",
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
        ax.set_title(dim, fontweight="bold", fontsize=10)
        ax.set_ylabel("Media punteggio (1-5)")
        ax.set_ylim(0, 5)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {Path(filename).resolve()}")


# =========================
# OPZIONE C — TABELLA DI SIGNIFICATIVITÀ
# =========================

def plot_significance_table(df_results_2, df_results_3, filename):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("off")

    table_data = []
    cell_colors = []

    # colori bianco/nero friendly
    COLOR_SIG = "#2c3e50"      # blu scuro per significativo
    COLOR_NOTSIG = "#ecf0f1"   # grigio chiaro per non significativo

    for dim in DIMENSIONS:
        row = [dim]
        colors = ["white"]

        for cond, df_r in [("Raw", df_results_2),
                           ("Refined", df_results_2),
                           ("Final", df_results_3)]:
            sub = df_r[(df_r["condizione"] == cond) & (df_r["dimensione"] == dim)]
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

    col_labels = ["Dimension", "Raw", "Refined (2-agent)", "Final (3-agent)"]
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

    # testo bianco per celle scure
    for i, row_data in enumerate(table_data):
        for j, val in enumerate(row_data):
            if cell_colors[i][j] == COLOR_SIG:
                table[i + 1, j].set_text_props(color="white", fontweight="bold")

    # intestazioni
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        "Friedman test significance per dimension and condition\n"
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
df3 = pd.read_csv(FILE_3AGENT)
res2 = pd.read_csv(RESULTS_2AGENT)
res3 = pd.read_csv(RESULTS_3AGENT)

# Bar chart
print("=== BAR CHART ===")
plot_barchart(df2, res2, "Raw",
              "Raw (Agent 1)",
              "rq3_barchart_raw.png")

plot_barchart(df2, res2, "Refined",
              "Refined (2-agent)",
              "rq3_barchart_refined.png")

plot_barchart(df3, res3, "Final",
              "Final (3-agent)",
              "rq3_barchart_final.png")

# Tabella significatività
print("\n=== TABELLA ===")
plot_significance_table(res2, res3,
                        "rq3_significance_table.png")