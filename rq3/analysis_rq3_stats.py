import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from pathlib import Path

# =========================
# CONFIG
# =========================

FILE_2AGENT = "risultati_analisi_completi_2agent.csv"
FILE_3AGENT = "risultati_finali_multiagente_3agent.csv"
OUTPUT_2AGENT = "rq3_friedman_2agent.csv"
OUTPUT_3AGENT = "rq3_friedman_3agent.csv"

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
# FRIEDMAN TEST
# =========================

def run_friedman(df, suffixes, label):
    results = []
    print(f"\n=== {label} ===")

    for suffix in suffixes:
        print(f"\n-- {suffix} --")
        for dim in DIMENSIONS:
            col = f"{dim}_{suffix}"
            if col not in df.columns:
                print(f"  Colonna {col} non trovata")
                continue

            # pivot: righe = frasi (Original_Row), colonne = lingue
            pivot = df.pivot_table(
                index="Original_Row",
                columns="Language",
                values=col,
                aggfunc="first"
            )

            pivot = pivot.reindex(columns=LANGUAGES).dropna()

            if len(pivot) < 2:
                print(f"  {dim}: frasi insufficienti ({len(pivot)})")
                continue

            groups = [pivot[lang].values for lang in LANGUAGES]
            stat, p = friedmanchisquare(*groups)

            results.append({
                "condizione": suffix,
                "dimensione": dim,
                "statistica": round(stat, 3),
                "p_value": round(p, 5),
                "significant": p < 0.05,
                "n_frasi": len(pivot),
                **{f"mean_{lang.replace(' ', '_').replace('(', '').replace(')', '')}":
                   round(pivot[lang].mean(), 3) for lang in LANGUAGES}
            })

            sig = "✓" if p < 0.05 else ""
            print(f"  {dim}: stat={stat:.3f}, p={p:.5f} {sig}")

    return pd.DataFrame(results)


# =========================
# 2 AGENTI
# =========================

df2 = pd.read_csv(FILE_2AGENT)
results_2 = run_friedman(df2, ["Raw", "Refined"], "2 AGENTI")
results_2.to_csv(OUTPUT_2AGENT, index=False, encoding="utf-8")
print(f"\nSalvato: {Path(OUTPUT_2AGENT).resolve()}")

# =========================
# 3 AGENTI
# =========================

df3 = pd.read_csv(FILE_3AGENT)
results_3 = run_friedman(df3, ["Raw", "Final"], "3 AGENTI")
results_3.to_csv(OUTPUT_3AGENT, index=False, encoding="utf-8")
print(f"\nSalvato: {Path(OUTPUT_3AGENT).resolve()}")