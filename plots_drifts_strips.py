import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from matplotlib.ticker import MultipleLocator


def dataset_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

def channel_to_col(channel):
    # Channel 1 -> Excel E -> pandas col 4
    return 3 + channel

def extract_channel_number(ch):
    """
    Extract numeric channel index from strings like 'ch1', 'ch01', ' ch12 ', etc.
    """
    match = re.search(r"\d+", str(ch))
    if match:
        return int(match.group())
    else:
        return None


def load_currents(filenames, skiprows, usecol, end_index):
    I_data = {}

    for i, fname in enumerate(filenames, start=1):
        df = pd.read_csv(fname, skiprows=skiprows, usecols=[usecol])
        df = df.fillna(0)

        I = df.iloc[:end_index, 0].to_numpy()
        I_data[f"I{i}"] = I

    return I_data

def load_currents_multichannel(filenames, skiprows, ch_start, ch_end, end_index):
    I_data = {}

    for fname in filenames:
        channels = range(ch_start, ch_end + 1)
        cols = [channel_to_col(ch) for ch in channels]

        df = pd.read_csv(fname, skiprows=skiprows, usecols=cols)
        df = df.fillna(0)

        for i, ch in enumerate(channels):
            I_data[f"ch{ch}"] = df.iloc[:end_index, i].to_numpy()

    return I_data


def find_spikes(I, n_sigma=4):
    median = np.median(I)
    mad = np.median(np.abs(I - median))
    sigma = 1.4826 * mad

    peak_indices = np.where(I < median - n_sigma * sigma)[0]
    return peak_indices

def build_mask(length, peak_indices, X):
    mask = np.ones(length, dtype=bool)

    for idx in peak_indices:
        start = max(0, idx - X)
        end   = min(length, idx + X + 1)
        mask[start:end] = False

    return mask

def offset_correct(I, mask):
    I_clean = I[mask]
    offset = np.mean(I_clean)
    I_clean_corr = I_clean - offset
    I_raw_corr = I - offset
    return I_clean_corr, I_raw_corr, offset

def drift_correct_src(I_bg_prev, I_src, I_bg_next, gap, t_int=1.0):
    """
    Fittet eine Gerade an bg_prev und bg_next (mit Lücke der Länge von src dazwischen)
    und zieht diese Gerade von src ab. 
    Zeitachse:
      bg_prev:  t = 0, 1, ..., N_prev-1
      src:      t = N_prev, N_prev+1, ..., N_prev+N_src-1  (Lücke im Fit)
      bg_next:  t = N_prev+N_src, ..., N_prev+N_src+N_next-1

    Von src[i] wird abgezogen: a * (N_prev + i) + b
    """
    N_prev = len(I_bg_prev)
    N_src  = len(I_src)
    N_next = len(I_bg_next)

    # Zeitachsen für bg_prev, src (Lücke) und bg_next
    t_prev = np.arange(N_prev, dtype=float)
    t_src = np.arange(N_prev+gap, N_prev + N_src + gap, dtype=float)
    t_next = np.arange(N_prev + N_src + 2*gap, N_prev + N_src + N_next + 2*gap, dtype=float)

    # Gemeinsamer Fit über beide bg-Segmente
    t_fit = np.concatenate([t_prev, t_next])
    I_fit = np.concatenate([I_bg_prev, I_bg_next])
    a, b = np.polyfit(t_fit, I_fit, 1)

    # Drift-Gerade auf src-Bereich auswerten und abziehen
    I_src_corr = I_src - (a * t_src + b)

    return I_src_corr, a, b


def compute_stats(I_clean_corr, offset):
    return {
        "mean": np.mean(I_clean_corr),   # ~ 0 (numerisch)
        "std": np.std(I_clean_corr),
        "offset": offset,
        "N_samples": len(I_clean_corr)
    }

def plot_raw_vs_clean(I, I_clean_corr, mask, t_int=1.0, end_index=None):
    if end_index is None:
        end_index = len(I)

    samples = np.arange(end_index) * t_int

    plt.figure(figsize=(7, 3))
    plt.plot(samples, I[:end_index], alpha=0.4, label="raw")
    plt.plot(samples[mask][:len(I_clean_corr)], I_clean_corr, label="cut + offset")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{title}_raw_vs_cut.png", dpi=300)
    plt.close()

def single_plot(I,title,savepath):
    plt.figure(figsize=(7, 3))
    plt.plot(I, label="Current")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{savepath}", dpi=300)
    plt.close()

def plot_raw_on_clean(
    I_raw,
    I_clean,
    mask,
    t_int=1.0,
    end_index=None,
    savepath=".",
    filename="plot.png",
    title=""
):
    # Zeitachsen
    t_raw = np.arange(len(I_raw)) * t_int
    t_clean = t_raw[mask]

    # optional: nur ersten Teil plotten
    if end_index is not None:
        tmax = end_index * t_int

        keep_raw = t_raw < tmax
        keep_clean = t_clean < tmax

        t_raw = t_raw[keep_raw]
        I_raw = I_raw[keep_raw]

        t_clean = t_clean[keep_clean]
        I_clean = I_clean[keep_clean]

    # Plot
    plt.figure(figsize=(7, 3))
    plt.plot(t_raw, I_raw, alpha=0.4, label="raw (offset corrected)")
    plt.plot(t_clean, I_clean, label="cleaned")

    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"{savepath}/{filename}.png", dpi=300)
    plt.close()

def concatenate_I_clean (I_data):
    I_clean_all = []
    for I in I_data.values():
        if do_cuts == 1:
            peak_indices = find_spikes(I, n_sigma)
            mask = build_mask(len(I), peak_indices, X)
        else:
            mask = np.ones(len(I), dtype=bool)
        I_data_clean = I[mask]
        I_clean_all.append(I_data_clean)

    I_all = np.concatenate(I_clean_all)-np.mean(I_data["I1"])
    return I_all

def analyze_channel_stability(
    stats_csv,
    output_csv="channel_stability_summary.csv"):
    
    df = pd.read_csv(stats_csv)

    # Gruppieren nach Channel
    summary = (
        df
        .groupby("channel")
        .agg(
            N_datasets=("dataset", "nunique"),
            mean_offset=("offset", "mean"),
            std_offset_between=("offset", "std"),
            mean_internal_std=("std", "mean"),
            mean_drift=("drift_slope_[A/s]", "mean")
        )
        .reset_index()
    )

    summary["channel_num"] = summary["channel"].str.replace("ch", "").astype(int)
    summary = summary.sort_values("channel_num").drop(columns="channel_num")

    # Leerzeile
    empty_row = pd.DataFrame([{
        "channel": "",
        "N_datasets": "",
        "mean_offset": "",
        "std_offset_between": "",
        "mean_internal_std": "",
        "mean_drift": ""
    }])
    
    # Average-Zeile
    avg_row = pd.DataFrame([{
        "channel": "Total Average",
        "N_datasets": summary["N_datasets"].iloc[0],  # gleich für alle
        "mean_offset": summary["mean_offset"].mean(),
        "std_offset_between": summary["std_offset_between"].mean(),
        "mean_internal_std": summary["mean_internal_std"].mean(),
        "mean_drift": summary["mean_drift"].mean()
    }])
    
    # Zusammenfügen
    summary = pd.concat([summary, empty_row, avg_row], ignore_index=True)

    summary.to_csv(output_csv, index=False)

    return summary

# ==================================================================================================
# ==================================================================================================
# ====================== MAIN ======================================================================
# ==================================================================================================
# ==================================================================================================


X = 0
n_sigma = 3
start_index = 15
end_index = 45
t_int = 1.0  # seconds
rows = []
gap = start_index + 15 
N_measurements = 5                      # Anzahl Messpaare (für Fehler des Mittelwerts: 1/N * sqrt(sum_k sigma_k^2))


# ==================================================================================================
single_or_multi = 1         # 0 = Single-Channel-Modus (HCC), 1 = Multi-Channel (Readout Strip Mode)
ch_start = 1                # erster Kanal (inkl.)
ch_end   = 64                # letzter Kanal (inkl.)
single_plots = 0            # 1 ja; 0 nein (WARNUNG: bei 1 werden 64 x N plots erstellt!)
plot_versus = 0
concatenate = 0 
do_cuts = 0
compute_signal = 1                      # Signal zwischen Background und Source berechnen
time_evolution_plots = 0                # 1 = Alle N Messungen eines Channels übereinander plotten (zeitliche Entwicklung)
channel_plots_src_with_bg_offset = 1    # 1 = Für src-Messungen den bg-Offset (channel-weise) abziehen statt eigenen Offset
clean_drifts = 1                        # 1 = Drift aus bg_prev + bg_next fitten und von src abziehen (ersetzt offset_correct für src)
drift_fit_plots = 0                     # 1 = Pro Channel einen Plot mit bg_prev/src/bg_next + Fit-Gerade
channel_time_evolution_plots = 0        # 1 = Pro Channel einen Plot mit allen bg/src Zeitverläufen übereinander
# ==================================================================================================

# Filenames
filenames = [
    "PATH/TO/FILE/BG1",     # bg1
    "PATH/TO/FILE/SRC1",    # src1
    "PATH/TO/FILE/BG1",     # bg2
    "PATH/TO/FILE/SRC1",    # src2
    "PATH/TO/FILE/BG1",     # bg3
]

# === Mapping: I1, I2, ... -> Dataset-Namen ===
# Wird automatisch aus filenames erstellt
measurement_map = {f"I{i}": dataset_name_from_path(f) for i, f in enumerate(filenames, start=1)}
# Ergebnis: {"I1": "260202_1247", "I2": "260202_1255", ...}

# === PAARE von Messungen (Hintergrund, Signal+Background) ====================================
pairs = [
    ("I1", "I2"),  # Background I1 (bg1), Signal+Background I2 (src1)
    ("I3", "I4"),  
    ("I5", "I6"),
    ("I7", "I8"),
    ("I9", "I10"),
    # ("I11", "I12"),
]
# letztes bg hier eintragen, sonst geht drift-gerade abzug nicht
extra_bg_next = "I11"
# ==================================================================================================

if single_or_multi == 0:
    # =======================================================================
    # ======================= SINGLE CHANNEL WORKFLOW =======================
    # =======================================================================

    I_data = load_currents(filenames, skiprows=57 + start_index, usecol=77, end_index=end_index)

    for name, I in I_data.items():
        if do_cuts == 1:
            peak_indices = find_spikes(I, n_sigma)
            mask = build_mask(len(I), peak_indices, X)
        else:
            mask = np.ones(len(I), dtype=bool)

        # offset correction
        I_clean_corr, I_raw_corr, offset = offset_correct(I, mask)

        # stats berechnen und speichern
        stats = compute_stats(I_clean_corr, offset)
        # Drift (linear slope)
        t = np.arange(len(I)) * t_int
        slope, intercept = np.polyfit(t, I, 1)

        rows.append({
            "measurement": name,
            "do_cuts": do_cuts,
            "X": X,

            # statistics
            "mean": stats["mean"],
            "std": stats["std"],
            "offset": stats["offset"],
            "N_samples": stats["N_samples"],

            # drift
            "drift_slope_[A/s]": slope
            })

        #plotten
        if plot_versus == 1:
            if do_cuts == 1:
                os.makedirs("raw_vs_cleaned_plots", exist_ok=True)
                plot_raw_on_clean(
                    I_raw=I_raw_corr,
                    I_clean=I_clean_corr,
                    mask=mask,
                    t_int=t_int,
                    end_index=end_index,
                    savepath="raw_vs_cleaned_plots",
                    filename=f"{name}_raw_vs_cleaned.png",
                    title=f"{name}")
        if single_plots == 1:
            os.makedirs("single_plots", exist_ok=True)
            if do_cuts == 1:
                title = f"{name}_cut"
            if do_cuts == 0:
                title = f"{name}_uncut"
            single_plot(
                I=I_clean_corr,
                title=title,
                savepath=f"single_plots/{title}.png")

        
    # ============= einzelne Messungen speichern =======================================================
    df_stats = pd.DataFrame(rows)
    # ==================================================================================================    


    # ============= Paare von Messungen auswerten =======================================================
    if compute_signal == 1 and pairs:
        rows_pairs = []

        for bg, src in pairs:

            offset_bg = df_stats.loc[
                df_stats["measurement"] == bg, "offset"
            ].values[0]

            offset_src = df_stats.loc[
                df_stats["measurement"] == src, "offset"
            ].values[0]

            signal = offset_src - offset_bg

            # signal nur bei quellenmessung eintragen (compute_signal = 1)
            df_stats.loc[
                df_stats["measurement"] == src, "signal"             
            ] = signal

            rows_pairs.append({
                "background": bg,
                "source": src,
                "offset_background": offset_bg,
                "offset_source": offset_src,
                "difference_(signaeft)": signal
            })

        df_pairs = pd.DataFrame(rows_pairs)
        df_pairs.to_csv("difference_signal_background.csv", index=False)
    # ===================================================================================================   

    # ==== ALL_STATS erst nach pairs-loop speichern, weil all_stats noch um das signal ergänzt wird =====
    df_stats.to_csv("all_stats.csv", index=False)

    # =================================================================================================== 

    # alle hintereinander plotten
    if concatenate == 1:
        I_clean_all = concatenate_I_clean(I_data)
        os.makedirs("messungen_concatenated", exist_ok=True)

        plt.plot(np.arange(len(I_clean_all)) * t_int, I_clean_all)
        plt.figure(figsize=(7, 3))
        plt.plot(np.arange(len(I_clean_all)) * t_int, I_clean_all, label="all, cleaned")
        plt.xlabel("Time [s]")
        plt.ylabel("Current [A]")
        plt.title("All measurements concatenated")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"messungen_concatenated/all_measurements_concatenated.png", dpi=300)
        plt.close()

    # Drift analysieren
    slopes = {}

    for name, I in I_data.items():
        t = np.arange(len(I)) * t_int
        a, b = np.polyfit(t, I, 1)
        slopes[name] = a

    for name, I in I_data.items():

        # Zeitachse
        t = np.arange(len(I)) * t_int

        # linearer Fit
        a, b = np.polyfit(t, I-np.mean(I), 1)

        # Fit-Gerade
        I_fit = a * t + b

        # Plot
        plt.figure(figsize=(7, 4))
        plt.plot(t, I-np.mean(I), alpha=0.5, label=f"{name} data")
        plt.plot(t, I_fit-np.mean(I_fit), color="black",
                label=f"linear fit: slope = {a:.2e} A/s")

        plt.xlabel("Time [s]")
        plt.ylabel("Current [A]")
        plt.title(f"{name}: linear drift fit")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"single_plots/{name}_linear_drift_fit.png", dpi=300)
        plt.close()

elif single_or_multi == 1:
    # =======================================================================
    # ======================= MULTI CHANNEL WORKFLOW ========================
    # =======================================================================

    rows = []   # (eine Zeile pro Kanal)

    for fname in filenames:

        # ---------- Datensatzname & Output-Ordner ----------
        dataset_name = dataset_name_from_path(fname)
        if single_plots == 1:    
            outdir = f"plots/{dataset_name}"
            os.makedirs(outdir, exist_ok=True)

        # ---------- Kanäle dieses Datensatzes laden ----------
        # IMMER alle 64 Channels laden für Statistik
        I_data = load_currents_multichannel(
            filenames=[fname],          
            skiprows=57 + start_index,
            ch_start=1,                 # Immer ab Channel 1
            ch_end=64,                  # Immer bis Channel 64
            end_index=end_index
        )

        # ---------- über alle Kanäle loopen ----------
        for ch_name, I in I_data.items():

            # ---------- Cuts / Mask ----------
            if do_cuts == 1:
                peak_indices = find_spikes(I, n_sigma)
                mask = build_mask(len(I), peak_indices, X)
            else:
                mask = np.ones(len(I), dtype=bool)

            # ---------- Offset-Korrektur ----------
            I_clean_corr, I_raw_corr, offset = offset_correct(I, mask)

            # ---------- Drift (linearer Fit auf ROHDATEN) ----------
            t = np.arange(len(I)) * t_int
            slope, intercept = np.polyfit(t, I, 1)

            # ---------- Statistik sammeln (für ALLE 64 Channels) ----------
            rows.append({
                "dataset": dataset_name,
                "channel": ch_name,
                "offset": offset,
                "std": np.std(I_clean_corr),
                "N_samples": len(I_clean_corr),
                "drift_slope_[A/s]": slope
            })

            # ---------- Plot (nur für ausgewählte Channels) ----------
            if single_plots == 1:
                # Extrahiere Channel-Nummer
                ch_num = int(ch_name.replace("ch", ""))
                
                # Nur plotten wenn Channel im gewünschten Bereich
                if ch_start <= ch_num <= ch_end:
                    title = f"{dataset_name} – {ch_name}"
                    savepath = f"{outdir}/{ch_name}.png"

                    print("PLOTTING:", savepath)
                    single_plot(
                        I=I_clean_corr,
                        title=title,
                        savepath=savepath
                    )

    # ---------- Statistik speichern ----------
    df_stats = pd.DataFrame(rows)
    df_stats.to_csv("all_stats_multichannel.csv", index=False)

    # ======= ZEITLICHE ENTWICKLUNG: Alle N Messungen pro Channel übereinander ==========
    if time_evolution_plots == 1:
        print("\n" + "="*70)
        print("ZEITLICHE ENTWICKLUNG: Erstelle Plots für alle Channels")
        print("="*70)
        
        # Output-Ordner erstellen
        time_outdir = "plots_time_evolution"
        os.makedirs(time_outdir, exist_ok=True)
        
        # Dictionary um alle Messungen pro Channel zu sammeln
        # Struktur: channel_data[ch_name] = [(dataset_name, I_clean_corr), ...]
        channel_data = {}
        
        # Alle Dateien nochmal durchgehen und Daten pro Channel sammeln
        for fname in filenames:
            dataset_name = dataset_name_from_path(fname)
            
            # Kanäle laden - nur die gewünschten für die Plots
            I_data = load_currents_multichannel(
                filenames=[fname],
                skiprows=57 + start_index,
                ch_start=ch_start,  
                ch_end=ch_end,
                end_index=end_index
            )
            
            # Für jeden Channel: Offset-korrigierte Daten speichern
            for ch_name, I in I_data.items():
                # Cuts / Mask
                if do_cuts == 1:
                    peak_indices = find_spikes(I, n_sigma)
                    mask = build_mask(len(I), peak_indices, X)
                else:
                    mask = np.ones(len(I), dtype=bool)
                
                # Offset-Korrektur
                I_clean_corr, I_raw_corr, offset = offset_correct(I, mask)
                
                # In Dictionary speichern
                if ch_name not in channel_data:
                    channel_data[ch_name] = []
                channel_data[ch_name].append((dataset_name, I_clean_corr))
        
        # Jetzt für jeden Channel einen Plot erstellen
        print(f"\nErstelle {len(channel_data)} Plots...")
        
        # Farben für die verschiedenen Messungen
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        for ch_name, measurements in channel_data.items():
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Alle Messungen dieses Channels plotten
            for i, (dataset_name, I_clean) in enumerate(measurements):
                t = np.arange(len(I_clean)) * t_int
                ax.plot(
                    t, 
                    I_clean, 
                    label=f"Measurement {i+1}",
                    color=colors[i % len(colors)],
                    alpha=0.7,
                    linewidth=1
                )
            
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Current [A]")
            ax.set_title(f"Electronics and cables: background current vs. time - {ch_name}")

            max_t = max(len(I_c) for _, I_c in measurements) * t_int
            ax.set_xlim(-0.5, max_t + 0.5)


            ax.legend(
                loc='lower left', 
                bbox_to_anchor=(0.1, -0.25, 0.8, 0.102), # (x, y, breite, höhe)
                ncol=len(filenames), 
                mode="expand", 
                borderaxespad=0.,
                frameon=False,
                fontsize=9
            )

            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{time_outdir}/{ch_name}_time_evolution.png", dpi=300)
            plt.close()
        
        print(f"[OK] Alle {len(channel_data)} Plots gespeichert in '{time_outdir}/'")

    # ======= ZEITLICHE ENTWICKLUNG: bg + src pro Channel, alle Paare in einem Plot ==========
    if channel_plots_src_with_bg_offset == 1 and pairs:
        print("\n" + "="*70)
        print("ZEITLICHE ENTWICKLUNG: bg + src (bg-offset korrigiert) pro Channel")
        print("="*70)

        src_outdir = "plots_after_corrected_drifts" if clean_drifts == 1 else "plots_time_evolution_src_bg_corrected"
        os.makedirs(src_outdir, exist_ok=True)
        # Unterordner für die channel-weisen Zeitverläufe
        ch_plot_subdir = os.path.join(src_outdir, "plots_time_evolution_src_bg_corrected")
        os.makedirs(ch_plot_subdir, exist_ok=True)

        # Farben: abwechselnd aus der Standard-Farbpalette, je Paar zwei aufeinanderfolgende
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        # Paar 1: colors[0]=bg1, colors[1]=src1 | Paar 2: colors[2]=bg2, colors[3]=src2 | ...
        pair_colors = [(colors[2*i % len(colors)], colors[(2*i+1) % len(colors)]) for i in range(len(pairs))]

        # Alle Daten vorher laden: pro Paar bg und src channel-weise
        # Struktur: all_pair_data[ch_name] = [(label, I_corr), ...]
        all_pair_data = {}
        rows_drift_signal = []   # für die drift-korrigierte Signal-CSV

        for pair_idx, (bg, src) in enumerate(pairs):
            bg_fname  = filenames[int(bg[1:]) - 1]
            src_fname = filenames[int(src[1:]) - 1]
            color_bg, color_src = pair_colors[pair_idx % len(pair_colors)]

            # bg laden
            I_bg_data = load_currents_multichannel(
                filenames=[bg_fname],
                skiprows=57 + start_index,
                ch_start=ch_start,
                ch_end=ch_end,
                end_index=end_index
            )
            # src laden
            I_src_data = load_currents_multichannel(
                filenames=[src_fname],
                skiprows=57 + start_index,
                ch_start=ch_start,
                ch_end=ch_end,
                end_index=end_index
            )

            # bg_next einmal pro Paar laden (wenn clean_drifts=1)
            I_bg_next_data = {}
            if clean_drifts == 1:
                pair_idx_next = pair_idx + 1
                if pair_idx_next < len(pairs):
                    bg_next_key = pairs[pair_idx_next][0]
                elif extra_bg_next is not None:
                    bg_next_key = extra_bg_next
                else:
                    bg_next_key = None
                    print(f"Kein bg_next für src{pair_idx+1} (letztes Paar hat kein folgendes bg) – src{pair_idx+1} wird übersprungen")

                if bg_next_key is not None:
                    bg_next_fname = filenames[int(bg_next_key[1:]) - 1]
                    I_bg_next_data = load_currents_multichannel(
                        filenames=[bg_next_fname],
                        skiprows=57 + start_index,
                        ch_start=ch_start,
                        ch_end=ch_end,
                        end_index=end_index
                    )

            for ch_name in I_bg_data:
                I_bg  = I_bg_data[ch_name]
                I_src = I_src_data.get(ch_name)

                # bg: eigener Offset abziehen
                if do_cuts == 1:
                    mask_bg = build_mask(len(I_bg), find_spikes(I_bg, n_sigma), X)
                else:
                    mask_bg = np.ones(len(I_bg), dtype=bool)
                bg_offset = np.mean(I_bg[mask_bg])
                I_bg_corr = I_bg - bg_offset

                # src: Drift-Korrektur ODER einfacher bg-Offset
                if I_src is not None:
                    if clean_drifts == 1 and I_bg_next_data:
                        I_bg_next = I_bg_next_data.get(ch_name)
                        if I_bg_next is not None:
                            I_src_corr, _a, _b = drift_correct_src(I_bg, I_src, I_bg_next, gap, t_int)
                        else:
                            I_src_corr = I_src - bg_offset  # Fallback: kein bg_next für diesen Channel
                    elif clean_drifts == 1 and not I_bg_next_data:
                        I_src_corr = None  # kein bg_next → src überspringen
                    else:
                        I_src_corr = I_src - bg_offset
                else:
                    I_src_corr = None

                if ch_name not in all_pair_data:
                    all_pair_data[ch_name] = []

                all_pair_data[ch_name].append((f"bg{pair_idx+1}",  I_bg_corr,  color_bg))
                if I_src_corr is not None:
                    all_pair_data[ch_name].append((f"src{pair_idx+1}", I_src_corr, color_src))
                    # Signal = mean(drift-korrigiertes src) für CSV
                    if clean_drifts == 1 and I_bg_next_data:
                        bg_next_key_for_csv = pairs[pair_idx + 1][0] if pair_idx + 1 < len(pairs) else (extra_bg_next if extra_bg_next is not None else None)
                        rows_drift_signal.append({
                            "src_file":    dataset_name_from_path(src_fname),
                            "bg_prev_file": dataset_name_from_path(bg_fname),
                            "bg_next_file": dataset_name_from_path(filenames[int(bg_next_key_for_csv[1:]) - 1]) if bg_next_key_for_csv else "",
                            "channel":     int(ch_name.replace("ch", "")),
                            "signal":      float(np.mean(I_src_corr)),
                        })

        # Einen Plot pro Channel
        channel_list = sorted(all_pair_data.keys(), key=lambda x: int(x.replace("ch", "")))

        if channel_time_evolution_plots == 1:
            print(f"\nErstelle {len(channel_list)} Plots...")
        for ch_name in channel_list if channel_time_evolution_plots == 1 else []:
            entries = all_pair_data[ch_name]

            fig, ax = plt.subplots(figsize=(10, 4))

            max_t = 0
            for label, I_corr, color in entries:
                t = np.arange(len(I_corr)) * t_int
                ax.plot(t, I_corr, label=label, color=color, alpha=0.8, linewidth=1)
                max_t = max(max_t, t[-1])

            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Current [A]")
            ax.set_title(f"bg & src (bg-offset corrected) – {ch_name}")
            ax.set_xlim(-0.5, max_t + 0.5)

            n_legend = len(entries)
            ax.legend(
                loc='lower left',
                bbox_to_anchor=(0.1, -0.25, 0.8, 0.102),
                ncol=n_legend,
                mode="expand",
                borderaxespad=0.,
                frameon=False,
                fontsize=9
            )

            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{ch_plot_subdir}/{ch_name}_bg_src_pairs.png", dpi=300, bbox_inches="tight")
            plt.close()

        print(f"[OK] Alle {len(channel_list)} Plots gespeichert in '{ch_plot_subdir}/'")

        # ======= DRIFT-FIT VISUALISIERUNG: bg_prev / src / bg_next mit Gerade ==========
        if clean_drifts == 1 and drift_fit_plots == 1:
            drift_fit_subdir = os.path.join(src_outdir, "plots_drift_fit")
            os.makedirs(drift_fit_subdir, exist_ok=True)

            print(f"\nErstelle Drift-Fit-Plots in '{drift_fit_subdir}/'...")

            for pair_idx, (bg, src) in enumerate(pairs):
                bg_fname  = filenames[int(bg[1:]) - 1]
                src_fname = filenames[int(src[1:]) - 1]

                # bg_next bestimmen
                if pair_idx + 1 >= len(pairs):
                    if extra_bg_next is not None:
                        bg_next_key = extra_bg_next
                    else:
                        print(f"Kein bg_next für src{pair_idx+1} – kein Drift-Fit-Plot")
                        continue
                else:
                    bg_next_key = pairs[pair_idx + 1][0]
                bg_next_fname = filenames[int(bg_next_key[1:]) - 1]

                I_bg_prev_all = load_currents_multichannel(filenames=[bg_fname],   skiprows=57+start_index, ch_start=ch_start, ch_end=ch_end, end_index=end_index)
                I_src_all     = load_currents_multichannel(filenames=[src_fname],  skiprows=57+start_index, ch_start=ch_start, ch_end=ch_end, end_index=end_index)
                I_bg_next_all = load_currents_multichannel(filenames=[bg_next_fname], skiprows=57+start_index, ch_start=ch_start, ch_end=ch_end, end_index=end_index)

                ch_list = sorted(I_bg_prev_all.keys(), key=lambda x: int(x.replace("ch", "")))

                for ch_name in ch_list:
                    I_bg_prev = I_bg_prev_all[ch_name]
                    I_src     = I_src_all.get(ch_name)
                    I_bg_next = I_bg_next_all.get(ch_name)

                    if I_src is None or I_bg_next is None:
                        continue

                    N_prev = len(I_bg_prev)
                    N_src  = len(I_src)
                    N_next = len(I_bg_next)

                    # Zeitachsen mit Lücken (identisch zur drift_correct_src Funktion)
                    t_prev = np.arange(N_prev, dtype=float)
                    t_src  = np.arange(N_prev + gap,             N_prev + gap + N_src,              dtype=float)
                    t_next = np.arange(N_prev + N_src + 2*gap,     N_prev + N_src + N_next + 2*gap,   dtype=float)

                    # Fit über bg_prev und bg_next
                    t_fit = np.concatenate([t_prev, t_next])
                    I_fit_data = np.concatenate([I_bg_prev, I_bg_next])
                    a, b = np.polyfit(t_fit, I_fit_data, 1)

                    # Gerade über gesamten Zeitbereich
                    t_all  = np.concatenate([t_prev, t_src, t_next])
                    t_line = np.linspace(t_all[0], t_all[-1], 500)
                    I_line = a * t_line + b

                    fig, ax = plt.subplots(figsize=(10, 4))

                    ax.plot(t_prev, I_bg_prev, color='#1f77b4', linewidth=1, alpha=0.8, label=f"Previous bg")
                    ax.plot(t_src,  I_src,     color='#ff7f0e', linewidth=1, alpha=0.8, label=f"Src measurement")
                    ax.plot(t_next, I_bg_next, color='#2ca02c', linewidth=1, alpha=0.8, label=f"Next bg")
                    ax.plot(t_line, I_line,    color='black',   linewidth=1.2, linestyle='--', label=f"Linear fit: a={a:.2e} A/s")

                    ax.set_xlabel("Time [s]")
                    ax.set_ylabel("Current [A]")
                    ax.set_title(f"Drift fit: Previous Background - Source - Next Background – {ch_name}")
                    ax.set_xlim(t_prev[0] - 2, t_next[-1] + 2)

                    ax.legend(
                        loc='lower left',
                        bbox_to_anchor=(0.1, -0.25, 0.8, 0.102),
                        ncol=4,
                        mode="expand",
                        borderaxespad=0.,
                        frameon=False,
                        fontsize=10
                    )
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{drift_fit_subdir}/pair{pair_idx+1}_{ch_name}_drift_fit.png", dpi=300, bbox_inches="tight")
                    plt.close()

            print(f"[OK] Drift-Fit-Plots gespeichert in '{drift_fit_subdir}/'")

        # Drift-korrigierte Signal-CSV speichern
        if clean_drifts == 1 and rows_drift_signal:
            df_drift_signal = pd.DataFrame(rows_drift_signal)
            df_drift_signal = df_drift_signal.sort_values(["src_file", "channel"]).reset_index(drop=True)
            drift_signal_csv = os.path.join(src_outdir, "signal_drift_corrected.csv")
            df_drift_signal.to_csv(drift_signal_csv, index=False)
            print(f"[OK] Drift-korrigierte Signal-CSV gespeichert: {drift_signal_csv}")

    # ======= SIGNAL BERECHNEN: Unterschied zwischen Background und Signal+Background =========
    if compute_signal == 1 and pairs:

        print("\n" + "="*70)
        print("SIGNAL-BERECHNUNG GESTARTET")
        print("="*70)

        # Ordner für Signal-Plots (bei clean_drifts=1: in plots_after_corrected_drifts)
        signal_plot_outdir = "plots_after_corrected_drifts" if clean_drifts == 1 else "."
        if clean_drifts == 1:
            os.makedirs(signal_plot_outdir, exist_ok=True)

        # === Datenquelle für Signal: drift-korrigierte CSV oder df_stats ===
        if clean_drifts == 1:
            drift_signal_csv = os.path.join("plots_after_corrected_drifts", "signal_drift_corrected.csv")
            if not os.path.exists(drift_signal_csv):
                print(f"\nFEHLER: {drift_signal_csv} nicht gefunden!")
                print("   Bitte zuerst channel_plots_src_with_bg_offset=1 und clean_drifts=1 ausführen.")
            else:
                df_drift = pd.read_csv(drift_signal_csv)
                # Umbauen in das Format das die Plot-Blöcke erwarten:
                # background=src_file, source=src_file, channel=chX, signal=signal
                rows_signal = []
                for pair_idx, (bg, src) in enumerate(pairs):
                    src_name = dataset_name_from_path(filenames[int(src[1:]) - 1])
                    pair_rows = df_drift[df_drift["src_file"] == src_name].copy()
                    if len(pair_rows) == 0:
                        print(f"Keine drift-korrigierten Daten für src{pair_idx+1} ({src_name})")
                        continue
                    for _, row in pair_rows.iterrows():
                        rows_signal.append({
                            "background": bg,
                            "source":     src,
                            "channel":    f"ch{int(row['channel'])}",
                            "signal":     row["signal"],
                        })

                if len(rows_signal) == 0:
                    print("\nFEHLER: Keine Signal-Daten aus drift-CSV!")
                else:
                    df_signal = pd.DataFrame(rows_signal)
                    df_signal.to_csv(os.path.join(signal_plot_outdir, "signal_multichannel_drift_corrected.csv"), index=False)
                    print(f"\n[OK] Signal aus drift-korrigierten Daten ({len(df_signal)} Zeilen)")

                    # ======= SIGNAL PLOTTEN für jedes Paar ==========
                    print("\n" + "="*70)
                    print("SIGNAL-PLOTS ERSTELLEN (drift-korrigiert)")
                    print("="*70)

                    for bg, src in pairs:
                        pair_data = df_signal[(df_signal["background"] == bg) & (df_signal["source"] == src)].copy()
                        if len(pair_data) == 0:
                            print(f"\nKeine Daten für Paar {bg} vs {src}")
                            continue
                        pair_data["channel_num"] = pair_data["channel"].str.replace("ch", "").astype(int)
                        pair_data = pair_data.sort_values("channel_num")
                        channels  = pair_data["channel_num"]
                        signals   = pair_data["signal"]

                        fig, ax = plt.subplots(figsize=(11, 4))
                        ax.plot(channels, signals, 'o-', markersize=4, linewidth=1)
                        ax.set_xlabel("Channel")
                        ax.set_ylabel("Signal [A]")
                        pair_num = pairs.index((bg, src)) + 1
                        ax.set_title(f"Signal per channel: bg{pair_num} vs src{pair_num} (drift corrected)")
                        ax.xaxis.set_major_locator(MultipleLocator(3))
                        ax.xaxis.set_minor_locator(MultipleLocator(1))
                        ax.grid(which="major", axis="x", linestyle="-", alpha=0.35)
                        ax.grid(which="minor", axis="x", linestyle=":", alpha=0.2)
                        ax.grid(which="major", axis="y", linestyle="-", alpha=0.3)
                        ax.set_xlim(0.5, 64.5)
                        plt.tight_layout()
                        filename = os.path.join(signal_plot_outdir, f"signal_bg{pair_num}_vs_src{pair_num}.png")
                        plt.savefig(filename, dpi=300)
                        plt.close()
                        print(f"  [OK] Plot gespeichert: {filename}")

                    print("\n" + "="*70)
                    print("FERTIG!")
                    print("="*70)

                    # ======= FINALER PLOT: Mean über alle Paare mit Errorbars ==========
                    print("\n" + "="*70)
                    print("FINALER PLOT: Mean-Signal über alle Paare (drift-korrigiert)")
                    print("="*70)

                    signal_summary = (
                        df_signal
                        .groupby("channel")
                        .agg(
                            mean_signal=("signal", "mean"),
                            std_signal=("signal",  "std"),
                            N_pairs=("signal",     "count")
                        )
                        .reset_index()
                    )
                    signal_summary["channel_num"] = signal_summary["channel"].str.replace("ch", "").astype(int)
                    signal_summary = signal_summary.sort_values("channel_num")
                    channels_plot  = signal_summary["channel_num"]

                    fig, ax = plt.subplots(figsize=(11, 4))
                    ax.errorbar(
                        channels_plot,
                        signal_summary["mean_signal"],
                        yerr=signal_summary["std_signal"],
                        fmt="o-", capsize=3, markersize=4, linewidth=1,
                        label=f"Mean signal over 5 measurements"
                    )
                    ax.set_xlabel("Channel")
                    ax.set_ylabel("Mean Signal [A]")
                    ax.set_title("Mean signal per channel, source left")
                    ax.xaxis.set_major_locator(MultipleLocator(3))
                    ax.xaxis.set_minor_locator(MultipleLocator(1))
                    ax.grid(which="major", axis="x", linestyle="-", alpha=0.35)
                    ax.grid(which="minor", axis="x", linestyle=":", alpha=0.2)
                    ax.grid(which="major", axis="y", linestyle="-", alpha=0.3)
                    ax.set_xlim(0.5, 64.5)
                    ax.legend(
                        loc='lower center',
                        bbox_to_anchor=(0.5, -0.28),
                        frameon=False,
                        fontsize=11
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(signal_plot_outdir, "signal_mean_all_pairs_with_errorbars.png"), dpi=300)
                    plt.close()
                    print(f"[OK] Finaler Plot gespeichert: {signal_plot_outdir}/signal_mean_all_pairs_with_errorbars.png")
                    signal_summary.drop(columns="channel_num").to_csv(
                        os.path.join(signal_plot_outdir, "signal_mean_summary.csv"), index=False)

                    # ======= AREA FINAL: N Einzelflächen + Mittelwert ==========
                    print("\n" + "="*70)
                    print("AREA FINAL: Einzelflächen mit Fehler aus 15s-Std pro Channel")
                    print("="*70)

                    rows_area = []
                    areas     = []
                    delta_areas = []

                    for pair_idx2, (bg2, src2) in enumerate(pairs):
                        # Signal pro Channel für dieses Paar
                        pd2 = df_signal[
                            (df_signal["background"] == bg2) & (df_signal["source"] == src2)
                        ].copy()
                        pd2["channel_num"] = pd2["channel"].str.replace("ch", "").astype(int)

                        # Std pro Channel aus all_stats_multichannel (15s-Rauschen der src-Messung)
                        src_dataset = measurement_map.get(src2, src2)
                        df_src_stats = df_stats[df_stats["dataset"] == src_dataset][["channel", "std"]].copy()
                        df_src_stats = df_src_stats.rename(columns={"std": "std_15s"})

                        # Mergen
                        pd2 = pd2.merge(df_src_stats, on="channel", how="left")

                        # Tote Channels ausschließen (signal==0 und std_15s==0)
                        mask = ~((pd2["signal"] == 0.0) & (pd2["std_15s"] == 0.0))
                        pd2 = pd2[mask]

                        # Fläche und Fehler
                        A_k     = pd2["signal"].sum()
                        dA_k    = np.sqrt((pd2["std_15s"] ** 2).sum())

                        areas.append(A_k)
                        delta_areas.append(dA_k)

                        rows_area.append({
                            "pair":    pair_idx2 + 1,
                            "src":     src2,
                            "area":    A_k,
                            "delta_area": dA_k,
                        })
                        print(f"  Paar {pair_idx2+1} ({src2}): A = {A_k:.4e} A  ±  {dA_k:.4e} A")

                    # Mittelwert und Gauss-Fehler des Mittelwerts
                    A_mean   = np.mean(areas)
                    dA_mean  = (1.0 / N_measurements) * np.sqrt(sum(da**2 for da in delta_areas))

                    print(f"\n  Mittelwert: A = {A_mean:.4e} A  ±  {dA_mean:.4e} A  (1/N * sqrt(sum dA_k^2), N={N_measurements})")

                    # Leerzeile + Mittelwert-Zeile anhängen
                    rows_area.append({"pair": "", "src": "", "area": "", "delta_area": ""})
                    rows_area.append({
                        "pair":       "mean",
                        "src":        f"N={N_measurements}",
                        "area":       A_mean,
                        "delta_area": dA_mean,
                    })

                    df_area_final = pd.DataFrame(rows_area)
                    area_final_path = os.path.join(signal_plot_outdir, "area_final.csv")
                    df_area_final.to_csv(area_final_path, index=False)
                    print(f"[OK] area_final.csv gespeichert: {area_final_path}")

                    # ======= ZUSÄTZLICHER PLOT: Alle Paare übereinander ==========
                    print("\n" + "="*70)
                    print("ZUSÄTZLICHER PLOT: Alle Paare übereinander (drift-korrigiert)")
                    print("="*70)

                    markers_list = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
                    colors_list  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                    fig, ax = plt.subplots(figsize=(11, 4))
                    for i, (bg, src) in enumerate(pairs):
                        pair_data = df_signal[(df_signal["background"] == bg) & (df_signal["source"] == src)].copy()
                        pair_data["channel_num"] = pair_data["channel"].str.replace("ch", "").astype(int)
                        pair_data = pair_data.sort_values("channel_num")
                        ax.plot(
                            pair_data["channel_num"], pair_data["signal"],
                            marker=markers_list[i % len(markers_list)],
                            linestyle='-', markersize=4, linewidth=1, alpha=0.7,
                            label=f"Measurement {i+1}",
                            color=colors_list[i % len(colors_list)]
                        )
                    ax.set_xlabel("Channel")
                    ax.set_ylabel("Signal [A]")
                    ax.set_title("Signal per channel - Source left")
                    ax.xaxis.set_major_locator(MultipleLocator(3))
                    ax.xaxis.set_minor_locator(MultipleLocator(1))
                    ax.grid(which="major", axis="x", linestyle="-", alpha=0.35)
                    ax.grid(which="minor", axis="x", linestyle=":", alpha=0.2)
                    ax.grid(which="major", axis="y", linestyle="-", alpha=0.3)
                    ax.set_xlim(0.5, 64.5)
                    ax.legend(
                        loc='lower left',
                        bbox_to_anchor=(0.1, -0.25, 0.8, 0.102),
                        ncol=len(pairs),
                        mode="expand",
                        borderaxespad=0.,
                        frameon=False
                    )
                    plt.tight_layout()
                    plt.savefig(os.path.join(signal_plot_outdir, "signal_all_pairs_overlayed.png"), dpi=300)
                    plt.close()
                    print(f"[OK] Overlay-Plot gespeichert: {signal_plot_outdir}/signal_all_pairs_overlayed.png")

                    # ======= GRUPPIERTE PLOTS: Channels addiert ==========
                    print("\n" + "="*70)
                    print("GRUPPIERTE PLOTS: Channel-Summen")
                    print("="*70)

                    # Gruppierungen: (n_groups, channels_per_group)
                    groupings = [
                        (8,  8),   #  8 Gruppen à  8 Channels
                        (16, 4),   # 16 Gruppen à  4 Channels
                        (32, 2),   # 32 Gruppen à  2 Channels
                    ]

                    grouped_outdir = os.path.join(signal_plot_outdir, "plots_grouped_channels")
                    os.makedirs(grouped_outdir, exist_ok=True)

                    # Für jede Gruppierung einen Mean- und einen Overlay-Plot
                    for n_groups, ch_per_group in groupings:
                        label = f"{n_groups}x{ch_per_group}"
                        print(f"\n→ Gruppierung {label}...")

                        # signal_summary bereits vorhanden (channel_num + mean_signal)
                        # Gruppe zuweisen: Channels 1-8 → Gruppe 1, 9-16 → Gruppe 2, ...
                        ss = signal_summary.copy()
                        ss["group"] = ((ss["channel_num"] - 1) // ch_per_group) + 1

                        # Pro Paar und Gruppe: Summe der Signale berechnen
                        rows_grouped = []
                        for pair_idx2, (bg2, src2) in enumerate(pairs):
                            pd2 = df_signal[(df_signal["background"] == bg2) & (df_signal["source"] == src2)].copy()
                            pd2["channel_num"] = pd2["channel"].str.replace("ch", "").astype(int)
                            pd2["group"] = ((pd2["channel_num"] - 1) // ch_per_group) + 1
                            grp_sums = pd2.groupby("group")["signal"].sum().reset_index()
                            grp_sums["pair_idx"] = pair_idx2 + 1
                            rows_grouped.append(grp_sums)

                        df_grouped = pd.concat(rows_grouped, ignore_index=True)

                        # Mean + Std über alle Paare pro Gruppe
                        grp_summary = (
                            df_grouped.groupby("group")
                            .agg(mean_signal=("signal", "mean"), std_signal=("signal", "std"), N=("signal", "count"))
                            .reset_index()
                        )

                        # Plot 1: Mean mit Errorbars
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.errorbar(
                            grp_summary["group"], grp_summary["mean_signal"],
                            yerr=grp_summary["std_signal"],
                            fmt="o", capsize=3, markersize=5, linewidth=1,
                            label=f"Mean signal ({ch_per_group} ch summed, N={grp_summary['N'].iloc[0]} pairs)"
                        )
                        # X-Tick-Labels: "1–8", "9–16", ... statt Gruppennummer
                        # Bei 32x2: nur die kleinere Channel-Nummer anzeigen, z.B. "1" für ch1+ch2
                        xtick_pos = list(range(1, n_groups + 1))
                        if ch_per_group == 2:
                            xtick_labels = [str((g-1)*ch_per_group+1) for g in xtick_pos]
                        else:
                            xtick_labels = [f"{(g-1)*ch_per_group+1}–{g*ch_per_group}" for g in xtick_pos]

                        ax.set_xlabel("Channels")
                        ax.set_ylabel("Summed Signal [A]")
                        ax.set_title(f"Signal grouped {label} – mean over all pairs (drift corrected)")
                        ax.set_xticks(xtick_pos)
                        ax.set_xticklabels(xtick_labels, fontsize=8)
                        ax.grid(which="major", axis="y", linestyle="-", alpha=0.3)
                        ax.grid(which="major", axis="x", linestyle="-", alpha=0.2)
                        ax.set_xlim(0.5, n_groups + 0.5)
                        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), frameon=False, ncol=3)
                        plt.tight_layout()
                        plt.savefig(os.path.join(grouped_outdir, f"signal_grouped_{label}_mean.png"), dpi=300)
                        plt.close()

                        # Plot 2: Alle Paare übereinander
                        fig, ax = plt.subplots(figsize=(8, 4))
                        for pair_idx2, (bg2, src2) in enumerate(pairs):
                            pair_grp = df_grouped[df_grouped["pair_idx"] == pair_idx2 + 1]
                            ax.plot(
                                pair_grp["group"], pair_grp["signal"],
                                marker=markers_list[pair_idx2 % len(markers_list)],
                                linestyle='-', markersize=5, linewidth=1, alpha=0.7,
                                label=f"Measurement {pair_idx2+1}",
                                color=colors_list[pair_idx2 % len(colors_list)]
                            )
                        ax.set_xlabel("Channels")
                        ax.set_ylabel("Summed Signal [A]")
                        ax.set_title(f"Signal grouped {label} – all pairs (drift corrected)")
                        ax.set_xticks(xtick_pos)
                        ax.set_xticklabels(xtick_labels, fontsize=8)
                        ax.grid(which="major", axis="y", linestyle="-", alpha=0.3)
                        ax.grid(which="major", axis="x", linestyle="-", alpha=0.2)
                        ax.set_xlim(0.5, n_groups + 0.5)
                        ax.legend(
                            loc='lower center',
                            bbox_to_anchor=(0.5, -0.35),
                            ncol=3,
                            columnspacing=2.0,
                            frameon=False,
                        )
                        fig.subplots_adjust(bottom=0.24)
                        plt.savefig(os.path.join(grouped_outdir, f"signal_grouped_{label}_all_pairs.png"), dpi=300)
                        plt.close()

                        print(f"  [OK] {label}: mean + overlay gespeichert")

                    print(f"[OK] Gruppierte Plots gespeichert in '{grouped_outdir}/'")

        else:
            # ======= FALLBACK: kein clean_drifts → alter Workflow mit offset_correct =========
            rows_signal = []
            for bg, src in pairs:
                bg_dataset  = measurement_map.get(bg, bg)
                src_dataset = measurement_map.get(src, src)
                df_bg  = df_stats[df_stats["dataset"] == bg_dataset]
                df_src = df_stats[df_stats["dataset"] == src_dataset]
                if len(df_bg) == 0 or len(df_src) == 0:
                    continue
                for ch in df_bg["channel"].unique():
                    offset_bg  = df_bg.loc[df_bg["channel"] == ch, "offset"].values[0]
                    offset_src = df_src.loc[df_src["channel"] == ch, "offset"].values[0]
                    rows_signal.append({
                        "background": bg, "source": src,
                        "channel": ch,
                        "offset_background": offset_bg,
                        "offset_source": offset_src,
                        "signal": offset_src - offset_bg
                    })
            if rows_signal:
                df_signal = pd.DataFrame(rows_signal)
                df_signal.to_csv("signal_multichannel.csv", index=False)
                print(f"\n[OK] Signal berechnet (offset-Methode): {len(df_signal)} Zeilen")

    # ======= Channel-Stabilität analysieren =========
    df_channel_summary = analyze_channel_stability(
        stats_csv="all_stats_multichannel.csv",
        output_csv="channel_stability_summary.csv")

    # Channel-Nummern extrahieren für x-Achse
    df = pd.read_csv("channel_stability_summary.csv")
    
    # Filter: Nur echte Channels (ch1, ch2, ...), keine "Total Average" oder leere Zeilen
    df = df[df["channel"].str.startswith("ch", na=False)].copy()

    # Channel-Nummern
    channels = df["channel"].str.replace("ch", "").astype(int)

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.errorbar(
        channels,
        df["mean_offset"],
        yerr=df["std_offset_between"],
        fmt="o",
        capsize=3,
        markersize=4,
        linewidth=1
    )

    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean offset [A]")
    ax.set_title(f'Electronics and cables: channel-wise mean offsets across {df["N_datasets"].iloc[0]} measurements')

    ax.legend(
        [f'Mean Offset ($\pm$ Std. Dev.), N = {df["N_datasets"].iloc[0]}'],
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.3), # (x, y, breite, höhe) 
        frameon=False)

    # X-Achse: Ticks & Grid
    ax.xaxis.set_major_locator(MultipleLocator(3))   # alle 3 Channels
    ax.xaxis.set_minor_locator(MultipleLocator(1))   # jeder Channel

    # Grid: fein + grob
    ax.grid(which="major", axis="x", linestyle="-", alpha=0.35)
    ax.grid(which="minor", axis="x", linestyle=":", alpha=0.2)
    ax.grid(which="major", axis="y", linestyle="-", alpha=0.3)

    ax.set_xlim(0.5, 64.5)

    plt.tight_layout()
    plt.savefig("mean_offset_with_errorbars.png", dpi=300)
    plt.close()
