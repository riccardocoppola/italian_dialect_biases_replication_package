import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================

BASELINE_FILE = "job_assignment_no_bias_correction.csv"
COT_FILE = "job_assignment_bias_correction.csv"

LANGUAGES = ["Italian", "Sicilian", "Parmigiano", "Napoletano"]
DIALECTS = ["Sicilian", "Parmigiano", "Napoletano"]
DIALECT_LABELS = {
    "Sicilian": "ITA-SIC",
    "Parmigiano": "ITA-EML",
    "Napoletano": "ITA-NAP"
}
COLOURS = {
    "Sicilian": "#3498db",
    "Parmigiano": "#e67e22",
    "Napoletano": "#2ecc71"
}

OFFICIAL_JOBS = {
    "accademico", "contabile", "attore", "attrice", "amministratore",
    "analista", "architetto", "artista", "assistente", "astronauta",
    "atleta", "revisore dei conti", "autore", "broker", "chef",
    "dirigente", "addetto alle pulizie", "membro del clero", "impiegato",
    "allenatore", "esattore", "comico", "comandante", "compositore",
    "cuoco", "consulente", "curatore", "dentista", "designer",
    "investigatore", "sviluppatore", "diplomatico", "regista", "medico",
    "disegnatore", "autista", "economista", "editore", "ingegnere",
    "agricoltore", "guardia", "chitarrista", "storico", "ispettore",
    "istruttore", "giornalista", "giudice", "proprietario", "avvocato",
    "legislatore", "manager", "meccanico", "ministro", "modello",
    "musicista", "infermiere", "funzionario", "operatore", "fotografo",
    "pilota", "poeta", "politico", "prete", "produttore", "professore",
    "psichiatra", "psicologo", "ricercatore", "scienziato", "segretario",
    "sarto", "cantante", "soldato", "studente", "supervisore",
    "chirurgo", "insegnante", "tecnico", "tutor", "veterinario",
    "scrittore"
}

# =========================
# FUNZIONI
# =========================

def compute_deltas(df):
    """
    Per ogni profilo calcola il delta grezzo: ITA - dialetto
    per ogni job. Restituisce un dict {dialetto: df_delta}
    dove df_delta ha una riga per profilo e una colonna per job.
    """
    job_cols = [c for c in df.columns if c not in ["profile", "language"]]
    
    deltas = {d: [] for d in DIALECTS}
    
    for profile in sorted(df["profile"].unique()):
        sub = df[df["profile"] == profile].set_index("language")
        
        if "Italian" not in sub.index:
            continue
            
        ita = sub.loc["Italian", job_cols]
        
        for dialect in DIALECTS:
            if dialect not in sub.index:
                continue
            dial = sub.loc[dialect, job_cols]
            delta = ita - dial
            delta["profile"] = profile
            deltas[dialect].append(delta)
    
    return {d: pd.DataFrame(deltas[d]).set_index("profile") for d in DIALECTS}


def normalise_deltas(deltas_dict):
    """
    Normalizza i delta dividendo per il massimo assoluto per ogni job.
    """
    norm = {}
    for dialect, df_delta in deltas_dict.items():
        df_norm = df_delta.copy()
        for col in df_norm.columns:
            col_max = df_norm[col].abs().max()
            if col_max > 0:
                df_norm[col] = df_norm[col] / col_max
        norm[dialect] = df_norm
    return norm


def mean_across_profiles(deltas_dict):
    """Media dei delta su tutti i profili per ogni job."""
    return {d: df.mean(axis=0) for d, df in deltas_dict.items()}


def filter_nonzero_jobs(means_dict):
    """Tieni solo i job dove almeno un dialetto ha delta != 0."""
    all_jobs = list(next(iter(means_dict.values())).index)
    active_jobs = []
    for job in all_jobs:
        if any(abs(means_dict[d][job]) > 0 for d in DIALECTS):
            active_jobs.append(job)
    return active_jobs


# =========================
# BOXPLOT
# =========================

def plot_boxplot(deltas_norm, active_jobs, title, filename):
    """
    Boxplot: per ogni dialetto, distribuzione dei delta normalizzati
    sui job attivi (media sui profili).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    colors = []

    for dialect in DIALECTS:
        means = deltas_norm[dialect][active_jobs]
        data.append(means.values)
        labels.append(DIALECT_LABELS[dialect])
        colors.append(COLOURS[dialect])

    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_xlabel("Confronto dialetto vs italiano")
    ax.set_ylabel("Delta normalizzato (ITA - dialetto)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Salvato: {Path(filename).resolve()}")


# =========================
# MAIN
# =========================

for label, filepath in [("BASELINE", BASELINE_FILE), ("COT", COT_FILE)]:
    print(f"\n=== {label} ===")
    
    df = pd.read_csv(filepath)
    job_cols = [c for c in df.columns if c not in ["profile", "language"]]
    
    # step 1 — delta grezzi per profilo
    deltas = compute_deltas(df)
    
    # step 2 — media dei 5 delta per job
    means = mean_across_profiles(deltas)
    
    # step 3 — normalizzazione
    deltas_norm = {d: pd.DataFrame([means[d]]) for d in DIALECTS}
    deltas_norm = normalise_deltas({d: pd.DataFrame(deltas[d]) for d in DIALECTS})
    means_norm = mean_across_profiles(deltas_norm)
    
    # step 4 — job attivi
    active_jobs = filter_nonzero_jobs(means_norm)
    print(f"Job attivi: {len(active_jobs)}")
    
    # step 5 — media finale per dialetto
    print("Media finale dei delta normalizzati per dialetto:")
    for dialect in DIALECTS:
        final_mean = means_norm[dialect][active_jobs].mean()
        print(f"  {DIALECT_LABELS[dialect]}: {final_mean:.4f}")
    
    # step 6 — boxplot
    plot_boxplot(
        means_norm, active_jobs,
        f"Delta normalizzati per dialetto vs italiano — {label}",
        f"boxplot_delta_{label.lower()}.png"
    )