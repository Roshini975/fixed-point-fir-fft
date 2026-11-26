"""
Python reference for the Fixed-Point FIR Filter and FFT project.

Generates a noisy test signal, designs a low-pass FIR filter, filters the
signal, and saves everything to text files. Your C code reads input.txt and
coeffs.txt, and its output is compared against reference_output.txt.

Run:  python3 fir_reference.py
"""
import numpy as np
from scipy import signal

# ---------------- Parameters ----------------
FS = 8000          # sampling rate (Hz)
N = 1024           # number of samples
F_LOW = 100        # wanted tone (Hz)   -> should pass
F_HIGH = 2000      # unwanted tone (Hz) -> should be removed
CUTOFF = 500       # low-pass cutoff (Hz)
NUM_TAPS = 63      # filter length (odd gives a symmetric linear-phase filter)
SEED = 42          # fixed seed so results are repeatable

# ---------------- 1. Generate the test signal ----------------
rng = np.random.default_rng(SEED)
t = np.arange(N) / FS
x = (0.5 * np.sin(2 * np.pi * F_LOW * t)
     + 0.3 * np.sin(2 * np.pi * F_HIGH * t)
     + 0.05 * rng.standard_normal(N))
x = np.clip(x, -0.99, 0.99)   # keep inside [-1, 1) so it fits Q15 later

# ---------- 2. Design the low-pass FIR filter (Hamming window) ----------
h = signal.firwin(NUM_TAPS, CUTOFF, fs=FS, window="hamming")

# ---------------- 3. Filter the signal ----------------
# lfilter with zero initial state is y[n] = sum_k h[k] * x[n-k],
# the same direct-form FIR loop you will write in C.
y = signal.lfilter(h, 1.0, x)

# ---------------- 4. Save files for the C program ----------------
np.savetxt("input.txt", x, fmt="%.9f")
np.savetxt("coeffs.txt", h, fmt="%.9f")
np.savetxt("reference_output.txt", y, fmt="%.9f")

# ---------------- 5. Quick checks ----------------
print(f"Samples: {N}, taps: {NUM_TAPS}, cutoff: {CUTOFF} Hz, fs: {FS} Hz")
print(f"Coefficient sum (should be about 1.0): {h.sum():.6f}")
print(f"Max |input|: {np.max(np.abs(x)):.4f}  Max |output|: {np.max(np.abs(y)):.4f}")
print(f"Max |coeff|: {np.max(np.abs(h)):.4f} (must be < 1 for Q15)")

# Reference FFT (use this later to check your C FFT)
X = np.fft.rfft(x)
Y = np.fft.rfft(y)
np.savetxt("reference_fft_input_mag.txt", np.abs(X), fmt="%.6f")
np.savetxt("reference_fft_output_mag.txt", np.abs(Y), fmt="%.6f")

# ---------------- 6. Plot (optional) ----------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freqs = np.fft.rfftfreq(N, 1 / FS)
    w, resp = signal.freqz(h, worN=2048, fs=FS)

    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    ax[0].plot(t[:200], x[:200], label="input (noisy)")
    ax[0].plot(t[:200], y[:200], label="filtered")
    ax[0].set_title("Time domain (first 200 samples)")
    ax[0].set_xlabel("Time (s)"); ax[0].legend()

    ax[1].plot(freqs, 20 * np.log10(np.abs(X) / N + 1e-12), label="input")
    ax[1].plot(freqs, 20 * np.log10(np.abs(Y) / N + 1e-12), label="filtered")
    ax[1].set_title("Spectrum")
    ax[1].set_xlabel("Frequency (Hz)"); ax[1].set_ylabel("dB"); ax[1].legend()

    ax[2].plot(w, 20 * np.log10(np.abs(resp) + 1e-12))
    ax[2].set_title("Filter frequency response")
    ax[2].set_xlabel("Frequency (Hz)"); ax[2].set_ylabel("dB")
    ax[2].set_ylim(-100, 5)

    plt.tight_layout()
    plt.savefig("reference_plot.png", dpi=120)
    print("Saved reference_plot.png")
except ImportError:
    print("matplotlib not installed; skipping plot (pip install matplotlib)")
