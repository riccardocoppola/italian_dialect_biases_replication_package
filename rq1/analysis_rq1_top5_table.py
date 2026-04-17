import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

BASELINE_FILE = "job_assignment_no_bias_correction.csv"
OUTPUT_FILE = "rq1_top5_jobs_table.png"

LANGUAGES = ["Italian", "Sicilian", "Parmigiano", "Napoletano"]
LANG_LABELS = {
    "Italian": "ITA",
    "Sicilian": "SIC",
    "Parmigiano": "EML",
    "Napoletano": "NAP"
}

# =========================
# MAIN
# =========================

df = pd.read_csv(BASELINE_FILE)
job_cols = [c for c in df.columns if c not in ["profile", "language"]]

rows = []
for profile in sorted(df["profile"].unique()):
    for lang in LANGUAGES:
        sub = df[(df["profile"] == profile) & (df["language"] == lang)]
        if sub.empty:
            continue
        top5 = sub[job_cols].iloc[0].sort_values(ascending=False).head(5)
        top5_jobs = ", ".join(top5.index.tolist())
        rows.append({
            "Profile": f"Profile {profile} — {LANG_LABELS[lang]}",
            "Top 5 jobs": top5_jobs
        })

table_df = pd.DataFrame(rows)

# =========================
# PLOT
# =========================

fig, ax = plt.subplots(figsize=(14, len(rows) * 0.6 + 1.5))
ax.axis("off")

table = ax.table(
    cellText=table_df.values,
    colLabels=table_df.columns,
    loc="center",
    cellLoc="left"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

# intestazioni nere
for j in range(len(table_df.columns)):
    table[0, j].set_facecolor("black")
    table[0, j].set_text_props(color="white", fontweight="bold")

# tutte le celle bianche
for i in range(len(rows)):
    for j in range(len(table_df.columns)):
        table[i + 1, j].set_facecolor("white")

ax.set_title("Top 5 jobs per profile and linguistic variety — Baseline",
             fontweight="bold", fontsize=13, pad=20)

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()
print(f"Salvato: {Path(OUTPUT_FILE).resolve()}")