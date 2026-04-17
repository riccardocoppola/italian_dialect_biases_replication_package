import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# =========================
# CONFIG
# =========================

INPUT_CSV = "rq2_adjectives_roles.csv"
MAPPING_CSV = "all_dialects_traslated.csv"
OUTPUT_CSV = "rq2_cochran_roles_results.csv"
MODEL = "gpt-4.1-mini"

ADJECTIVES = [
    "sporca", "pulita", "rumorosa", "fredda", "affettuosa",
    "ritardataria", "pia", "onesta", "furba", "precisa",
    "divertente", "noiosa", "educata", "maleducata", "inesperta",
    "colta", "sensibile"
]

LANGUAGES = ["italiano", "napoletano", "parmigiano", "siciliano"]

ROLES = [
    "giudice imparziale",
    "giudice imparziale corto",
    "uomo di paese",
    "uomo di paese corto"
]

# =========================
# COCHRAN'S Q TEST
# =========================

def cochran_q(data):
    data = np.array(data, dtype=float)
    n, k = data.shape
    row_sums = data.sum(axis=1)
    col_sums = data.sum(axis=0)
    total = data.sum()

    numerator = (k - 1) * (k * np.sum(col_sums ** 2) - total ** 2)
    denominator = k * total - np.sum(row_sums ** 2)

    if denominator == 0:
        return np.nan, np.nan

    Q = numerator / denominator
    p = 1 - stats.chi2.cdf(Q, df=k - 1)
    return Q, p

# =========================
# LOAD MAPPING
# =========================

mapping = pd.read_csv(MAPPING_CSV)
mapping.columns = mapping.columns.str.strip()

frase_to_id = {}
for _, row in mapping.iterrows():
    fid = row["File Name"]
    frase_to_id[row["Neapolitan Text"].strip()]       = fid
    frase_to_id[row["Automated Translation"].strip()] = fid
    frase_to_id[row["Parmigiano"].strip()]            = fid
    frase_to_id[row["Siciliano"].strip()]             = fid

print(f"Frasi mappate: {len(frase_to_id)}")

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(INPUT_CSV)
df = df[df["modello"] == MODEL]
df["frase"] = df["frase"].str.strip()
df["frase_id"] = df["frase"].map(frase_to_id)

print(f"Righe caricate: {len(df)}")
print(f"Frasi con ID: {df['frase_id'].notna().sum()}")
print(f"Frasi senza ID: {df['frase_id'].isna().sum()}")

df = df[df["frase_id"].notna()]

# =========================
# TEST PER OGNI RUOLO E AGGETTIVO
# =========================

results = []

for ruolo in ROLES:
    df_role = df[df["ruolo"] == ruolo]
    print(f"\nRuolo: {ruolo} — righe: {len(df_role)}")

    for aggettivo in ADJECTIVES:
        try:
            pivot = df_role.pivot_table(
                index="frase_id",
                columns="linguaggio",
                values=aggettivo,
                aggfunc="first"
            )

            pivot = pivot.reindex(columns=LANGUAGES).dropna()

            if len(pivot) < 2:
                print(f"  Skipped {aggettivo}: frasi insufficienti ({len(pivot)})")
                continue

            Q, p = cochran_q(pivot.values)
            props = pivot.mean()

            results.append({
                "ruolo": ruolo,
                "aggettivo": aggettivo,
                "Q": round(Q, 3),
                "p_value": round(p, 5),
                "significant": p < 0.05,
                "n_frasi": len(pivot),
                **{f"prop_{lang}": round(props[lang], 3) for lang in LANGUAGES}
            })

        except Exception as e:
            print(f"  Errore {aggettivo}: {e}")

# =========================
# SALVA RISULTATI
# =========================

results_df = pd.DataFrame(results).sort_values(["ruolo", "p_value"])
results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"\nRisultati salvati: {Path(OUTPUT_CSV).resolve()}")
print(f"\n=== RISULTATI ===")
print(results_df.to_string(index=False))