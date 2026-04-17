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

N_COMPARISONS = len(DIMENSIONS)  # 6 — per correzione Bonferroni

# =========================
# FRIEDMAN TEST
# =========================

def run_friedman(df, suffixes, label):
    results = []
    print(f"\n=== {label} ===")

    for suffix in suffixes:
        print(f"\n-- {suffix} --")
        
        # prima passata: calcola tutti i p-value grezzi
        raw_results = []
        for dim in DIMENSIONS:
            col = f"{dim}_{suffix}"
            if col not in df.columns:
                print(f"  Colonna {col} non trovata")
                continue

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

            raw_results.append({
                "condizione": suffix,
                "dimensione": dim,
                "statistica": round(stat, 3),
                "p_value_raw": p,
                "n_frasi": len(pivot),
                **{f"mean_{lang.replace(' ', '_').replace('(', '').replace(')', '')}":
                   round(pivot[lang].mean(), 3) for lang in LANGUAGES}
            })

        # seconda passata: applica correzione Bonferroni
        for r in raw_results:
            p_corrected = min(r["p_value_raw"] * N_COMPARISONS, 1.0)
            r["p_value_corrected"] = round(p_corrected, 5)
            r["p_value_raw"] = round(r["p_value_raw"], 5)
            r["significant"] = p_corrected < 0.05

            sig = "✓" if r["significant"] else ""
            print(f"  {r['dimensione']}: stat={r['statistica']:.3f}, "
                  f"p_raw={r['p_value_raw']:.5f}, "
                  f"p_corrected={r['p_value_corrected']:.5f} {sig}")

            results.append(r)

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