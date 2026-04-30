import pandas as pd
import numpy as np

df = pd.read_csv("job_assignment_no_bias_correction_with_deltas.csv")

for delta in ["delta_NAP", "delta_PAR", "delta_SIC"]:
    rows = df[df["language"] == delta]
    job_cols = [c for c in df.columns if c not in ["profile", "language"]]
    values = rows[job_cols].values.flatten()
    values = values[~np.isnan(values) & (values != 0)]
    print(f"{delta}: mean={values.mean():.3f}, max={values.max():.3f}, min={values.min():.3f}")