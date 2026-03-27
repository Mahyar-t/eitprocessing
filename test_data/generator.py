from numpy import size
import numpy as np
import matplotlib.pyplot as plt
import os


def generate_realistic_eit_signal(
    duration=60,
    fs=50,
    mean_bpm=14,
    bpm_variation=2.0,
    heart_rate_bpm=72,
    base_impedance=10.0,
    tidal_amplitude=0.9,
    cardiac_amplitude=0.04,
    drift_amplitude=0.18,
    noise_std=0.02,
    seed=42
):
    """
    Generate a more realistic synthetic EIT breathing signal.

    Parameters
    ----------
    duration : float
        Total signal duration in seconds.
    fs : float
        Sampling frequency in Hz.
    mean_bpm : float
        Mean breathing rate in breaths per minute.
    bpm_variation : float
        Breath-to-breath variability in breathing rate.
    heart_rate_bpm : float
        Heart rate in beats per minute for cardiac oscillation.
    base_impedance : float
        Baseline impedance level.
    tidal_amplitude : float
        Mean amplitude of the breathing-related impedance variation.
    cardiac_amplitude : float
        Amplitude of cardiac oscillation.
    drift_amplitude : float
        Amplitude of slow baseline drift.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    t : ndarray
        Time vector.
    signal : ndarray
        Synthetic EIT signal.
    eeli_baseline : ndarray
        Slowly varying end-expiratory baseline trend.
    breathing_only : ndarray
        Breathing component only.
    cardiac_only : ndarray
        Cardiac component only.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    n = len(t)

    # -----------------------------
    # 1) Slow EELI / baseline drift
    # -----------------------------
    # Combination of very low-frequency components
    drift = (
        drift_amplitude * 0.6 * np.sin(2 * np.pi * 0.008 * t) +
        drift_amplitude * 0.4 * np.sin(2 * np.pi * 0.003 * t + 1.2)
    )
    eeli_baseline = base_impedance + drift

    # -----------------------------------------
    # 2) Breath-to-breath timing and amplitude
    # -----------------------------------------
    breathing_only = np.zeros_like(t)

    current_time = 0.0
    breath_starts = []

    while current_time < duration:
        # Random breathing rate for this breath
        this_bpm = max(6, mean_bpm + rng.normal(0, bpm_variation))
        cycle_duration = 60.0 / this_bpm

        # Random tidal amplitude for this breath
        this_amp = tidal_amplitude * (1 + rng.normal(0, 0.12))
        this_amp = max(0.3, this_amp)

        # Slightly variable inhale fraction
        inhale_fraction = np.clip(0.38 + rng.normal(0, 0.04), 0.25, 0.55)
        inhale_duration = cycle_duration * inhale_fraction
        exhale_duration = cycle_duration - inhale_duration

        breath_starts.append(current_time)

        idx = (t >= current_time) & (t < current_time + cycle_duration)
        tau = t[idx] - current_time

        # Build an asymmetric breath:
        # - faster inspiration
        # - slower expiration
        breath_wave = np.zeros_like(tau)

        insp_idx = tau < inhale_duration
        exp_idx = ~insp_idx

        # Inspiration: smooth rise from 0 to 1
        if np.any(insp_idx):
            x = tau[insp_idx] / inhale_duration
            breath_wave[insp_idx] = 0.5 * (1 - np.cos(np.pi * x))

        # Expiration: smooth decay from 1 back to 0
        if np.any(exp_idx):
            x = (tau[exp_idx] - inhale_duration) / exhale_duration
            breath_wave[exp_idx] = 0.5 * (1 + np.cos(np.pi * x))

        breathing_only[idx] = this_amp * breath_wave
        current_time += cycle_duration

    # --------------------------------
    # 3) Cardiac-related oscillations
    # --------------------------------
    heart_rate_hz = heart_rate_bpm / 60.0

    # Small modulation so the cardiac signal is not perfectly uniform
    cardiac_envelope = 1.0 + 0.15 * np.sin(2 * np.pi * 0.02 * t + 0.8)

    cardiac_only = cardiac_amplitude * cardiac_envelope * (
        0.75 * np.sin(2 * np.pi * heart_rate_hz * t) +
        0.25 * np.sin(2 * np.pi * 2 * heart_rate_hz * t + 0.5)
    )

    # -----------------------------
    # 4) Add noise
    # -----------------------------
    noise = rng.normal(0, noise_std, size=n)

    # -----------------------------
    # 5) Final signal
    # -----------------------------
    signal = eeli_baseline + breathing_only + cardiac_only + noise

    return t, signal, eeli_baseline, breathing_only, cardiac_only


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    t, signal, eeli_baseline, breathing_only, cardiac_only = generate_realistic_eit_signal(
        duration=60,
        fs=50,
        mean_bpm=14,
        bpm_variation=2.0,
        heart_rate_bpm=72,
        base_impedance=10.0,
        tidal_amplitude=0.9,
        cardiac_amplitude=0.04,
        drift_amplitude=0.18,
        noise_std=0.02,
        seed=42
    )

    # Save plot
    save_path = os.path.join(os.getcwd(), "test_data")

    # Save raw signal as float32 binary
    signal.astype(np.float32).tofile(os.path.join(save_path, "realistic_eit_signal.bin"))

    plt.figure(figsize=(12, 5.5))
    plt.plot(t, signal, label="Synthetic realistic EIT signal")
    plt.plot(t, eeli_baseline, "--", label="Slow EELI baseline drift")
    plt.xlabel("Time (s)")
    plt.ylabel("Relative impedance (a.u.)")
    plt.title("More Realistic Synthetic EIT Breathing Signal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "realistic_eit_signal_plot.png"), dpi=200)
    # plt.show()