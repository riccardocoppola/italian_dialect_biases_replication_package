import json

with open("job_assignment_no_bias_correction.json") as f:
    data = json.load(f)

DISCARDED = {"dj", "DJ", "street artist", "club manager", "bartender",
             "storcio", "manutentore", "muratore", "impresario",
             "carpentiere", "ballerino", "terapeuta", "nuotatore",
             "fisioterapista"}

count = 0
for record in data:
    for lang in ["Italian", "Napoletano", "Parmigiano", "Sicilian"]:
        jobs = [j.strip() for j in record.get(f"jobs_{lang}", "").split(",")]
        count += sum(1 for j in jobs if j in DISCARDED)

print(f"Job eliminati: {count} su {len(data) * 4 * 5} totali")