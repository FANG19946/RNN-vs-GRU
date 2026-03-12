import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# configuration
# ---------------------------------

INPUT_DIR = "."
OUTPUT_DIR = "report_plots"

folders = {
    "Q1": "Q1_gradient_hist",
    "Q2": "Q2_saturation_hist",
    "Q3": "Q3_gradient_norm",
    "Q4": "Q4_gate_saturation",
    "Q5": "Q5_rho"
}

EXTRA_FOLDER = os.path.join(OUTPUT_DIR, "EXTRA_credit")

# ---------------------------------
# helper
# ---------------------------------

def _log10_grad(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]

    if v.size == 0:
        return np.array([]), 0

    pos = v > 0
    zero_frac = 100.0 * float((~pos).mean())

    floor = np.log10(np.finfo(np.float64).tiny)

    out = np.full(v.shape, floor)
    if np.any(pos):
        out[pos] = np.log10(v[pos])

    return out, zero_frac


# ---------------------------------
# create directories
# ---------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EXTRA_FOLDER, exist_ok=True)

for f in folders.values():
    os.makedirs(os.path.join(OUTPUT_DIR, f), exist_ok=True)

# ---------------------------------
# find npz files
# ---------------------------------

npz_files = [
    f for f in os.listdir(INPUT_DIR)
    if f.endswith("_final_state.npz")
]

print("Found files:")
for f in npz_files:
    print(" ", f)

# ---------------------------------
# process files
# ---------------------------------

for file in npz_files:

    path = os.path.join(INPUT_DIR, file)
    z = np.load(path)

    name = file.replace("_final_state.npz","")

    print("\nProcessing:", name)

    is_extra = name.startswith("EXTRA")

    # =====================================================
    # Q1 gradient histogram
    # =====================================================

    if "grad_time" in z:

        g_last = z["grad_time"][-1]
        g_last = g_last[np.isfinite(g_last)]

        g = g_last
        source = "final checkpoint"

        if g_last.size > 0 and np.unique(g_last).size <= 2:
            g_all = z["grad_time"]
            g_all = g_all[np.isfinite(g_all)]
            if g_all.size > 0:
                g = g_all
                source = "all checkpoints"

        g_log, zero_frac = _log10_grad(g)

        if g_log.size > 0:

            plt.figure()
            plt.hist(g_log, bins=60)

            plt.title(f"log10||dL/dh_t||\n{name}\n{source}, zero={zero_frac:.1f}%")
            plt.xlabel("log10 gradient")
            plt.ylabel("count")

            if is_extra:
                savepath = os.path.join(EXTRA_FOLDER, name+"_gradient_hist.png")
            else:
                savepath = os.path.join(OUTPUT_DIR, folders["Q1"], name+"_gradient_hist.png")

            plt.savefig(savepath)
            plt.close()

    # =====================================================
    # Q2 saturation histogram
    # =====================================================

    if "sat_time" in z:

        s = z["sat_time"][-1]
        s = s[np.isfinite(s)]

        plt.figure()
        plt.hist(s, bins=60, range=(0,1))

        plt.title(f"Hidden saturation distance\n{name}")
        plt.xlabel("distance to saturation")
        plt.ylabel("count")

        if is_extra:
            savepath = os.path.join(EXTRA_FOLDER, name+"_saturation_hist.png")
        else:
            savepath = os.path.join(OUTPUT_DIR, folders["Q2"], name+"_saturation_hist.png")

        plt.savefig(savepath)
        plt.close()

    # =====================================================
    # Q3 gradient norm
    # =====================================================

    if "gradient_norm" in z:

        grad_norm = z["gradient_norm"]

        plt.figure()
        plt.plot(grad_norm)

        plt.title(f"Gradient norm during training\n{name}")
        plt.xlabel("iteration")
        plt.ylabel("gradient norm")

        if is_extra:
            savepath = os.path.join(EXTRA_FOLDER, name+"_gradient_norm.png")
        else:
            savepath = os.path.join(OUTPUT_DIR, folders["Q3"], name+"_gradient_norm.png")

        plt.savefig(savepath)
        plt.close()

    # =====================================================
    # Q4 gate saturation (GRU only)
    # =====================================================

    if "gate_z_sat_time" in z:

        zsat = z["gate_z_sat_time"][-1]
        zsat = zsat[np.isfinite(zsat)]

        plt.figure()
        plt.hist(zsat, bins=60, range=(0,1))

        plt.title(f"GRU update gate saturation\n{name}")
        plt.xlabel("distance to saturation")

        savepath = os.path.join(OUTPUT_DIR, folders["Q4"], name+"_z_gate_hist.png")

        plt.savefig(savepath)
        plt.close()

    if "gate_r_sat_time" in z:

        rsat = z["gate_r_sat_time"][-1]
        rsat = rsat[np.isfinite(rsat)]

        plt.figure()
        plt.hist(rsat, bins=60, range=(0,1))

        plt.title(f"GRU reset gate saturation\n{name}")
        plt.xlabel("distance to saturation")

        savepath = os.path.join(OUTPUT_DIR, folders["Q4"], name+"_r_gate_hist.png")

        plt.savefig(savepath)
        plt.close()

    # =====================================================
    # Q5 spectral radius
    # =====================================================

    if "rho_Whh" in z:

        rho = z["rho_Whh"]

        plt.figure()
        plt.plot(rho)

        plt.title(f"Spectral radius rho(W_hh)\n{name}")
        plt.xlabel("checkpoint")
        plt.ylabel("rho")

        if is_extra:
            savepath = os.path.join(EXTRA_FOLDER, name+"_rho.png")
        else:
            savepath = os.path.join(OUTPUT_DIR, folders["Q5"], name+"_rho.png")

        plt.savefig(savepath)
        plt.close()

print("\nAll plots saved in:", OUTPUT_DIR)