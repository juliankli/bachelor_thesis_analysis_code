import numpy as np
import pandas as pd
import os
import re

# === Dateien ===
# Skript erwartet Dateien im selben Ordner wie das Skript selbst
# Format: signal_mean_summaryXXX.csv, wobei XXX die Cath-Zahl ist (50, 75, 100, ...)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Exakte Cathode-Spannungen (Nominal -> Exakt)
cath_exact = {
    50:  50.1,
    75:  75.1,
    100: 100.0,
    150: 153.2,
    200: 199.9,
    300: 301.6,
    400: 399.5,
    600: 599.8,
}

# Alle passenden CSV-Dateien finden (area_final_XXX.csv)
pattern = re.compile(r"area_final_(\d+)\.csv$")
files = []
for fname in os.listdir(script_dir):
    m = pattern.match(fname)
    if m:
        files.append((int(m.group(1)), os.path.join(script_dir, fname)))

# Nach Cath-Zahl sortieren
files.sort(key=lambda x: x[0])

if not files:
    print("⚠️  Keine area_final_*.csv Dateien gefunden!")
    exit()

print(f"Gefundene Dateien: {[f[1] for f in files]}")
print("="*70)

rows = []

for cath, fpath in files:
    df = pd.read_csv(fpath)

    # Nur Zeilen mit echten Paaren (pair ist eine Zahl), keine Leer- oder Mean-Zeile
    df_pairs = df[pd.to_numeric(df["pair"], errors="coerce").notna()].copy()
    df_pairs["area"]       = pd.to_numeric(df_pairs["area"],       errors="coerce")
    df_pairs["delta_area"] = pd.to_numeric(df_pairs["delta_area"], errors="coerce")
    df_pairs = df_pairs.dropna(subset=["area", "delta_area"])

    # Tote Einträge ausschließen (area==0 und delta_area==0)
    df_pairs = df_pairs[~((df_pairs["area"] == 0.0) & (df_pairs["delta_area"] == 0.0))]

    # N aus der Mean-Zeile auslesen (steht als "N=3" in der src-Spalte)
    mean_row = df[df["pair"] == "mean"]
    if len(mean_row) > 0:
        src_val = str(mean_row["src"].values[0])  # z.B. "N=3"
        N = int(src_val.split("=")[1])
        area       = float(mean_row["area"].values[0])
        delta_area = float(mean_row["delta_area"].values[0])
    else:
        # Fallback: selbst berechnen
        N          = len(df_pairs)
        area       = df_pairs["area"].mean()
        delta_area = (1.0 / N) * np.sqrt((df_pairs["delta_area"] ** 2).sum())

    print(f"[{cath:4d}]  Area = {area:.4e} A  |  delta = {delta_area:.4e} A  (N={N}, {len(df_pairs)} Paare)")

    cath_v = cath_exact.get(cath, float(cath))
    rows.append({
        "Mesh":        cath_v,
        "Area":        area,
        "delta_area":  delta_area,
    })

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# === Ergebnis speichern ===
df_result = pd.DataFrame(rows)
out_path = os.path.join(script_dir, "area_under_curves.csv")
df_result.to_csv(out_path, index=False)

print("\n" + "="*70)
print(f"✓ Ergebnis gespeichert: {out_path}")
print("="*70)
print(df_result.to_string(index=False))

# === Plot: Area vs Cath mit std als Fehlerbalken ===
fig, ax = plt.subplots(figsize=(8, 4))

ax.errorbar(
    df_result["Mesh"],
    df_result["Area"],
    yerr=df_result["delta_area"],
    fmt="o-",
    capsize=4,
    markersize=5,
    linewidth=1,
    label="Integrated Signal"
)

ax.set_xlabel("Cathode voltage [V]")
ax.set_ylabel("Integrated Signal [A]")
ax.set_title("Integrated signal for different cathode voltages")


# Ticks an glatten Zahlen, aber Datenpunkte an exakten Spannungen
nominal_voltages = sorted(cath_exact.keys())
ax.set_xticks(nominal_voltages)
ax.set_xticklabels([str(v) for v in nominal_voltages], fontsize=9)
ax.set_xlim(45, 605)
ax.grid(which="major", axis="y", linestyle="-", alpha=0.3)
ax.grid(which="major", axis="x", linestyle="-", alpha=0.3)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), frameon=False)

plt.tight_layout()
plot_path = os.path.join(script_dir, "amp_analysis.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"✓ Plot gespeichert: {plot_path}")
