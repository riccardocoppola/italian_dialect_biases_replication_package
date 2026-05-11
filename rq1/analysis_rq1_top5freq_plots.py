import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

CSV_FILE = "job_assignment_no_bias_correction.csv"

LANGUAGES = ["Italian", "Napoletano", "Parmigiano", "Sicilian"]
LANG_LABELS = {
    "Italian": "ITA",
    "Napoletano": "NEA",
    "Parmigiano": "PAR",
    "Sicilian": "SIC"
}
COLOURS = {
    "Italian":    "#a6cee3",
    "Napoletano": "#1f78b4",
    "Parmigiano": "#b2df8a",
    "Sicilian":   "#33a02c"
}

JOB_TRANSLATIONS = {
    "meccanico": "mechanic",
    "agricoltore": "farmer",
    "autista": "driver",
    "supervisore": "supervisor",
    "operatore": "operator",
    "assistente": "assistant",
    "allenatore": "coach",
    "atleta": "athlete",
    "attore": "actor",
    "modello": "model",
    "pilota": "pilot",
    "istruttore": "instructor",
    "attrice": "actress",
    "musicista": "musician",
    "cuoco": "cook",
    "fotografo": "photographer",
    "scrittore": "writer",
    "chef": "chef",
    "artista": "artist",
    "consulente": "consultant",
    "professore": "professor",
    "storico": "historian",
    "manager": "manager",
    "amministratore": "administrator",
    "comico": "comedian",
    "modella": "model",
    "dirigente": "executive",
    "revisore dei conti": "auditor",
    "addetto alle pulizie": "cleaner",
    "cantante": "singer",
    "regista": "director",
    "investigatore": "investigator",
    "soldato": "soldier",
    "comandante": "commander",
}

df = pd.read_csv(CSV_FILE)
job_cols = [c for c in df.columns if c not in ["profile", "language"]]

for profile in sorted(df["profile"].unique()):
    sub = df[df["profile"] == profile]

    mean_counts = sub[job_cols].mean()
    top5_jobs = mean_counts.sort_values(ascending=False).head(5).index.tolist()
    top5_labels = [JOB_TRANSLATIONS.get(j, j) for j in top5_jobs]

    x = np.arange(5)
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, lang in enumerate(LANGUAGES):
        lang_row = sub[sub["language"] == lang]
        if lang_row.empty:
            continue
        values = [lang_row[j].values[0] if j in lang_row.columns else 0
                  for j in top5_jobs]
        ax.bar(
            x + offsets[i] * width,
            values,
            width=width,
            color=COLOURS[lang],
            label=LANG_LABELS[lang],
            alpha=0.85
        )

    ax.set_xticks(x)
    ax.set_xticklabels(top5_labels, rotation=15, ha="right", fontsize=18)
    ax.set_ylabel("Frequency (out of 30 runs)", fontsize=22)
    ax.set_title(
        f"Top 5 most assigned jobs — Profile {profile}",
        fontweight="bold",
        fontsize=22
    )
    ax.tick_params(axis="y", labelsize=18)
    ax.legend(title="Variety", fontsize=16, title_fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    filename = f"rq1_barchart_top5avg_profile{profile}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {Path(filename).resolve()}")