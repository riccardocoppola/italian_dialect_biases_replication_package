import pandas as pd
import numpy as np

df = pd.read_csv("job_assignment_no_bias_correction_with_deltas.csv")
job_cols = [c for c in df.columns if c not in ["profile", "language"]]

for delta in ["delta_NAP", "delta_PAR", "delta_SIC"]:
    print(f"\n=== {delta} ===")
    for profile in ["1", "2", "3", "4", "5"]:
        row = df[(df["profile"].astype(str) == profile) & (df["language"] == delta)]
        if row.empty:
            continue
        values = row[job_cols].values.flatten()
        values = values[~np.isnan(values) & (values != 0)]
        print(f"  P{profile}: mean={values.mean():.3f}, max={values.max():.3f}, min={values.min():.3f}")