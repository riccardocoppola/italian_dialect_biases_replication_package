import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# CONFIG
# =========================

BASELINE_FILE = "job_assignment_no_bias_correction.csv"
COT_FILE = "job_assignment_bias_correction.csv"

OUTPUT_BASELINE = "job_assignment_no_bias_correction_with_deltas.csv"
OUTPUT_COT = "job_assignment_bias_correction_with_deltas.csv"

# =========================
# MAIN
# =========================

def process(input_file, output_file):
    df = pd.read_csv(input_file)
    job_cols = [c for c in df.columns if c not in ["profile", "language"]]

    # =========================
    # STEP 2 — delta per profilo
    # =========================
    delta_rows = []

    for profile in sorted(df["profile"].unique()):
        sub = df[df["profile"] == profile].set_index("language")

        if "Italian" not in sub.index:
            continue

        ita = sub.loc["Italian", job_cols]

        for dialect, label in [("Sicilian", "delta_SIC"), ("Parmigiano", "delta_EML"), ("Napoletano", "delta_NAP")]:
            if dialect not in sub.index:
                continue
            dial = sub.loc[dialect, job_cols]
            delta = dial - ita
            row = {"profile": profile, "language": label}
            row.update(delta.to_dict())
            delta_rows.append(row)

    df_deltas = pd.DataFrame(delta_rows)

    # =========================
    # STEP 3 — media dei 5 delta per confronto
    # =========================
    mean_rows = []
    for label in ["delta_SIC", "delta_EML", "delta_NAP"]:
        sub = df_deltas[df_deltas["language"] == label][job_cols]
        mean = sub.mean()
        row = {"profile": "mean", "language": f"mean_{label}"}
        row.update(mean.to_dict())
        mean_rows.append(row)

    df_means = pd.DataFrame(mean_rows)

    # =========================
    # STEP 4 — media di tutti i job per ogni mean_delta
    # =========================
    global_mean_rows = []
    for label in ["mean_delta_SIC", "mean_delta_EML", "mean_delta_NAP"]:
        sub = df_means[df_means["language"] == label][job_cols]
        global_mean = sub.mean(axis=1).values[0]
        row = {"profile": "global_mean", "language": f"global_{label}"}
        row.update({job: global_mean for job in job_cols})
        global_mean_rows.append(row)

    df_global_means = pd.DataFrame(global_mean_rows)

    # =========================
    # STEP 5 — concatena tutto
    # =========================
    df_full = pd.concat([df, df_deltas, df_means, df_global_means], ignore_index=True)

    # =========================
    # STEP 6 — normalizzazione per colonna
    # =========================
    for job in job_cols:
        col_max = df_full[job].abs().max()
        if col_max > 0:
            df_full[job] = df_full[job] / col_max

    # =========================
    # SALVA
    # =========================
    df_full.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Salvato: {Path(output_file).resolve()}")
    print(f"Righe totali: {len(df_full)}")
    print(df_full[["profile", "language"]].to_string())

print("=== BASELINE ===")
process(BASELINE_FILE, OUTPUT_BASELINE)

print("\n=== COT ===")
process(COT_FILE, OUTPUT_COT)