Replication package for the paper: “Measuring Linguistic Bias in LLMs Across I.N.P.S. (Italian, Neapolitan, Parmesan and Sicilian) Dialect Corpora”

## Overview

This repository contains the data, scripts and figures for three experiments investigating linguistic bias in GPT-4.1 mini across Standard Italian and three regional dialects (Neapolitan, Parmesan, Sicilian) using the matched guise probing technique.

## Repository Structure

### “rq1” — Experiment 1: Job Assignment

Scripts:
“analysis.py”: Data processing and χ² statistical analysis
“analysis_rq1_delta.py”: Delta computation and normalisation
“analysis_rq1_plots.py”: Boxplots and barcharts
“analysis_rq1_top5_table.py”: Table of top 5 most assigned occupations
“analysis_rq1_top5freq_plots.py: Barcharts of top 5 most assigned occupations by average frequency

Data:
“job_assignment_no_bias_correction.json”: Raw model outputs (30 runs × 5 profiles × 4 varieties)
“job_assignment_no_bias_correction.csv: Aggregated occupation frequencies per profile and variety
“job_assignment_no_bias_correction_with_deltas.csv”: Occupation frequencies with normalised delta values (dialect minus Italian)
“profile_statistical_results.csv: χ² test results and Cramér's V per profile

### “rq2“ — Experiment 2: Trait Adjective Attribution

Scripts:
“analysis_rq2_baseline.py”: Data processing for baseline condition
“analysis_rq2_roles.py”: Data processing for role prompting conditions
“analysis_rq2_cochran_baseline.py”: Cochran's Q test — baseline
“analysis_rq2_cochran_roles.py”: Cochran's Q test — role prompting conditions
“analysis_rq2_plots.py”: Barcharts for all conditions

Data:
“all_dialects_traslated.csv”: Dataset of 141 sentences in all four linguistic varieties
“rq2_adjectives_baseline.csv”: Binary response matrix — baseline condition
“rq2_adjectives_roles.csv: Binary response matrix — role prompting conditions
“rq2_cochran_baseline_results.csv”: Cochran's Q results — baseline
“rq2_cochran_roles_results.csv”: Cochran's Q results — role prompting conditions

### “rq3” — Experiment 3: Character Scoring

Scripts:
“analysis_rq3_stats.py”: Friedman test with Bonferroni correction
“analysis_rq3.py”: Data processing and radar plot generation
“analysis_rq3_plots.py”: Significance table and barcharts

Data:
“risultati_analisi_completi_2agent.csv: Raw and Refined scores for all sentences and varieties (2-step pipeline)
“rq3_friedman_2agent.csv”: Friedman test results with Bonferroni correction (Raw and Refined)

## How to Reproduce

All scripts are written in Python 3. Install dependencies with:

```bash
pip install pandas numpy matplotlib scipy
```

Run the scripts in each folder in the following order:

**RQ1:**
```bash
cd rq1
python analysis.py
python analysis_rq1_delta.py
python analysis_rq1_plots.py
python analysis_rq1_top5_table.py
python analysis_rq1_top5freq_plots.py
```

**RQ2:**
```bash
cd rq2
python analysis_rq2_baseline.py
python analysis_rq2_roles.py
python analysis_rq2_cochran_baseline.py
python analysis_rq2_cochran_roles.py
python analysis_rq2_plots.py
```

**RQ3:**
```bash
cd rq3
python analysis_rq3_stats.py
python analysis_rq3.py
python analysis_rq3_plots.py
```

## Dataset

The sentences used in Experiments 2 and 3 are drawn from the [Neapolitan Spoken Corpus](https://huggingface.co/datasets/anonymous-nsc-author/Neapolitan-Spoken-Corpus) and translated into Standard Italian, Parmesan and Sicilian using the tools described in the paper.



