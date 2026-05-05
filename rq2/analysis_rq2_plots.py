import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

BASELINE_RESULTS = "rq2_cochran_baseline_results.csv"
ROLES_RESULTS = "rq2_cochran_roles_results.csv"

LANGUAGES = ["italiano", "napoletano", "parmigiano", "siciliano"]
LANG_LABELS = {
    "italiano": "ITA",
    "napoletano": "NAP",
    "parmigiano": "PAR",
    "siciliano": "SIC"
}
COLOURS = {
    "italiano": "#95a5a6",
    "napoletano": "#2ecc71",
    "parmigiano": "#e67e22",
    "siciliano": "#3498db"
}

ADJECTIVE_TRANSLATIONS = {
    "sporca": "dirty",
    "pulita": "clean",
    "rumorosa": "noisy",
    "fredda": "cold",
    "affettuosa": "affectionate",
    "ritardataria": "tardy",
    "pia": "pious",
    "onesta": "honest",
    "furba": "cunning",
    "precisa": "precise",
    "divertente": "funny",
    "noiosa": "boring",
    "educata": "polite",
    "maleducata": "rude",
    "inesperta": "inexperienced",
    "colta": "educated",
    "sensibile": "sensitive"
}

# =========================
# BAR CHART — TUTTI GLI AGGETTIVI
# =========================

def plot_barchart(df, title, filename):
    df = df.copy()
    df["adjective_en"] = df["aggettivo"].map(ADJECTIVE_TRANSLATIONS).fillna(df["aggettivo"])
    df = df.sort_values("Q", ascending=False, na_position="last")

    adjectives = df["aggettivo"].tolist()
    adjectives_en = df["adjective_en"].tolist()

    x = np.arange(len(adjectives))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(adjectives) * 0.9), 6))

    for i, lang in enumerate(LANGUAGES):
        col = f"prop_{lang}"
        if col not in df.columns:
            continue
        values = df[col].values
        ax.bar(x + i * width, values, width,
               label=LANG_LABELS[lang],
               color=COLOURS[lang],
               alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(adjectives_en, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Proportion of Yes responses")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {Path(filename).resolve()}")


# =========================
# DOT PLOT CON Q STATISTIC
# =========================

def plot_dotplot(df, title, filename):
    df = df.copy().sort_values("Q", ascending=True, na_position="first")
    df = df[df["Q"].notna()]
    df["adjective_en"] = df["aggettivo"].map(ADJECTIVE_TRANSLATIONS).fillna(df["aggettivo"])

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))

    for _, row in df.iterrows():
        q = row["Q"]
        adj = row["adjective_en"]
        sig = row["significant"]

        color = "#e74c3c" if sig else "#95a5a6"
        size = 80 if sig else 40

        ax.scatter(q, adj, color=color, s=size, zorder=3)
        if not np.isnan(q):
            ax.hlines(adj, 0, q, color=color, lw=1.5, alpha=0.6)

    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Q statistic (Cochran's Q)")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=10, label="Significant (p < 0.05)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6",
               markersize=7, label="Not significant")
    ]
    ax.legend(handles=handles, loc="lower right")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {Path(filename).resolve()}")


# =========================
# MAIN
# =========================

# --- BASELINE ---
print("=== BASELINE ===")
df_base = pd.read_csv(BASELINE_RESULTS)

plot_barchart(df_base,
              "Proportion of Yes responses per adjective — Baseline",
              "rq2_barchart_baseline.png")

plot_dotplot(df_base,
             "Q statistic per adjective — Baseline",
             "rq2_dotplot_baseline.png")

# --- RUOLI ---
print("\n=== RUOLI ===")
df_roles = pd.read_csv(ROLES_RESULTS)

for ruolo in df_roles["ruolo"].unique():
    print(f"\nRuolo: {ruolo}")
    df_r = df_roles[df_roles["ruolo"] == ruolo]
    label = ruolo.replace(" ", "_")

    plot_barchart(df_r,
                  f"Proportion of Yes responses per adjective — {ruolo}",
                  f"rq2_barchart_{label}.png")

    plot_dotplot(df_r,
                 f"Q statistic per adjective — {ruolo}",
                 f"rq2_dotplot_{label}.png")