import json
import re
import pandas as pd
from pathlib import Path

# =========================
# CONFIG
# =========================

ROLE_FILE = "results_role_prompting.jsonl"
OUTPUT_CSV = "rq2_adjectives_roles.csv"

ADJECTIVES = [
    "sporca", "pulita", "rumorosa", "fredda", "affettuosa",
    "ritardataria", "pia", "onesta", "furba", "precisa",
    "divertente", "noiosa", "educata", "maleducata", "inesperta",
    "colta", "sensibile"
]

# =========================
# HELPERS
# =========================

def extract_adjective(prompt):
    match = re.search(r'è (\w+)\.$', prompt.strip())
    if match:
        return match.group(1)
    return None

def extract_phrase(prompt):
    match = re.search(r"dice\s*:?'(.+?)'\s*\n", prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def normalize_language(lang):
    if lang == "dialetto":
        return "napoletano"
    return lang

def parse_response(response):
    r = response.strip().lower()
    if r in ["sì", "si", "yes"]:
        return True
    return False

# =========================
# LOAD ROLE PROMPTING
# =========================

records = []
with open(ROLE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        aggettivo = extract_adjective(obj["prompt"])
        if aggettivo not in ADJECTIVES:
            continue
        frase = extract_phrase(obj["prompt"])
        records.append({
            "frase": frase,
            "aggettivo": aggettivo,
            "linguaggio": normalize_language(obj["language"]),
            "ruolo": obj.get("role", ""),
            "modello": obj.get("model_name", ""),
            "risposta": parse_response(obj["response"])
        })

df = pd.DataFrame(records)
print(f"Record caricati: {len(df)}")
print(f"Aggettivi trovati: {df['aggettivo'].unique()}")
print(f"Lingue trovate: {df['linguaggio'].unique()}")
print(f"Ruoli trovati: {df['ruolo'].unique()}")
print(f"Modelli trovati: {df['modello'].unique()}")
print(f"\nFrasi con None: {df['frase'].isna().sum()}")

# =========================
# PIVOT
# =========================

df_pivot = df.pivot_table(
    index=["frase", "linguaggio", "ruolo", "modello"],
    columns="aggettivo",
    values="risposta",
    aggfunc="first"
).reset_index()

df_pivot.columns.name = None

# riordina le colonne degli aggettivi nell'ordine della lista
cols = ["frase", "linguaggio", "ruolo", "modello"] + ADJECTIVES
df_pivot = df_pivot[cols]

# =========================
# SALVA
# =========================

df_pivot.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"\nCSV salvato: {Path(OUTPUT_CSV).resolve()}")
print(f"Righe: {len(df_pivot)}, Colonne: {len(df_pivot.columns)}")