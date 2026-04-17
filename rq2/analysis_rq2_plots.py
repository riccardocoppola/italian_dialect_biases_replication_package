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
    "parmigiano": "EML",
    "siciliano": "SIC"
}
COLOURS = {
    "italiano": "#95a5a6",
    "napoletano": "#2ecc71",
    "parmigiano": "#e67e22",
    "siciliano": "#3498db"
}

# =========================
# OPZIONE B — BAR CHART PER AGGETTIVO SIGNIFICATIVO
# =========================

def plot_barchart(df, title, filename):
    # filtra solo aggettivi significativi
    sig = df[df["significant"] == True].copy()

    if sig.empty:
        print(f"Nessun aggettivo significativo per {title}")
        return

    sig = sig.sort_values("Q", ascending=False)
    adjectives = sig["aggettivo"].tolist()

    x = np.arange(len(adjectives))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(adjectives) * 0.8), 6))

    for i, lang in enumerate(LANGUAGES):
        col = f"prop_{lang}"
        values = sig[col].values
        ax.bar(x + i * width, values, width,
               label=LANG_LABELS[lang],
               color=COLOURS[lang],
               alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(adjectives, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Proporzione di Sì")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {Path(filename).resolve()}")


# =========================
# OPZIONE C — DOT PLOT CON Q STATISTIC
# =========================

def plot_dotplot(df, title, filename):
    df = df.copy().sort_values("Q", ascending=True, na_position="first")
    df = df[df["Q"].notna()]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))

    for _, row in df.iterrows():
        q = row["Q"]
        adj = row["aggettivo"]
        sig = row["significant"]

        color = "#e74c3c" if sig else "#95a5a6"
        marker = "o" if sig else "o"
        size = 80 if sig else 40

        ax.scatter(q, adj, color=color, s=size, zorder=3)
        if not np.isnan(q):
            ax.hlines(adj, 0, q, color=color, lw=1.5, alpha=0.6)

    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Q statistic (Cochran's Q)")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # legenda manuale
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
               markersize=10, label="Significativo (p < 0.05)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6",
               markersize=7, label="Non significativo")
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
              "Proporzione Sì per aggettivo significativo — Baseline",
              "rq2_barchart_baseline.png")

plot_dotplot(df_base,
             "Q statistic per aggettivo — Baseline",
             "rq2_dotplot_baseline.png")

# --- RUOLI ---
print("\n=== RUOLI ===")
df_roles = pd.read_csv(ROLES_RESULTS)

for ruolo in df_roles["ruolo"].unique():
    print(f"\nRuolo: {ruolo}")
    df_r = df_roles[df_roles["ruolo"] == ruolo]

    label = ruolo.replace(" ", "_")

    plot_barchart(df_r,
                  f"Proporzione Sì per aggettivo significativo — {ruolo}",
                  f"rq2_barchart_{label}.png")

    plot_dotplot(df_r,
                 f"Q statistic per aggettivo — {ruolo}",
                 f"rq2_dotplot_{label}.png")