from __future__ import annotations
import os

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# Format specification from the Draeger loader
# ==================================================
FORMAT_SPECS = {
    "original": {
        "frame_size": 4358,
        "n_medibus_fields": 52,
    },
    "pressure_pod": {
        "frame_size": 4382,
        "n_medibus_fields": 58,
    },
}


@dataclass
class SyntheticDraegerData:
    format_name: str
    frame_size: int
    sample_frequency: float
    time: np.ndarray                # shape: (n_frames,), seconds
    pixel_impedance: np.ndarray     # shape: (n_frames, 32, 32), float32
    medibus_data: np.ndarray        # shape: (n_medibus_fields, n_frames), float32
    min_max_flags: np.ndarray       # shape: (n_frames,), int32
    event_markers: np.ndarray       # shape: (n_frames,), int32
    event_texts: list[str]          # len = n_frames, each <= 30 bytes after encoding
    timing_errors: np.ndarray       # shape: (n_frames,), int32
    unused_float32: np.ndarray      # shape: (n_frames,), float32


# ==================================================
# 1) Realistic synthetic breathing signal
# ==================================================
def generate_realistic_eit_signal(
    duration: float = 60.0,
    fs: float = 20.0,
    mean_bpm: float = 14.0,
    bpm_variation: float = 2.0,
    heart_rate_bpm: float = 72.0,
    base_impedance: float = 10.0,
    tidal_amplitude: float = 0.9,
    cardiac_amplitude: float = 0.04,
    drift_amplitude: float = 0.18,
    noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a realistic synthetic breathing-related impedance waveform.

    Returns
    -------
    time_seconds : ndarray
        Time vector in seconds.
    signal : ndarray
        Synthetic EIT-like global impedance waveform.
    """
    rng = np.random.default_rng(seed)
    time_seconds = np.arange(0, duration, 1 / fs)

    # slow baseline drift
    drift = (
        drift_amplitude * 0.6 * np.sin(2 * np.pi * 0.008 * time_seconds)
        + drift_amplitude * 0.4 * np.sin(2 * np.pi * 0.003 * time_seconds + 1.2)
    )

    # breath-to-breath variability
    breathing = np.zeros_like(time_seconds)
    current_time = 0.0

    while current_time < duration:
        this_bpm = max(6.0, mean_bpm + rng.normal(0, bpm_variation))
        cycle_duration = 60.0 / this_bpm

        this_amp = max(0.3, tidal_amplitude * (1.0 + rng.normal(0, 0.12)))

        inhale_fraction = np.clip(0.38 + rng.normal(0, 0.04), 0.25, 0.55)
        inhale_duration = cycle_duration * inhale_fraction
        exhale_duration = cycle_duration - inhale_duration

        idx = (time_seconds >= current_time) & (time_seconds < current_time + cycle_duration)
        tau = time_seconds[idx] - current_time
        wave = np.zeros_like(tau)

        insp_idx = tau < inhale_duration
        exp_idx = ~insp_idx

        # inspiration
        if np.any(insp_idx):
            x = tau[insp_idx] / inhale_duration
            wave[insp_idx] = 0.5 * (1 - np.cos(np.pi * x))

        # expiration
        if np.any(exp_idx):
            x = (tau[exp_idx] - inhale_duration) / exhale_duration
            wave[exp_idx] = 0.5 * (1 + np.cos(np.pi * x))

        breathing[idx] = this_amp * wave
        current_time += cycle_duration

    # cardiac oscillation
    heart_rate_hz = heart_rate_bpm / 60.0
    cardiac = cardiac_amplitude * (
        0.75 * np.sin(2 * np.pi * heart_rate_hz * time_seconds)
        + 0.25 * np.sin(2 * np.pi * 2 * heart_rate_hz * time_seconds + 0.5)
    )

    # measurement noise
    noise = rng.normal(0, noise_std, size=time_seconds.shape)

    signal = base_impedance + drift + breathing + cardiac + noise
    return time_seconds, signal


# ==================================================
# 2) Spatial map for 32x32 EIT-like frames
# ==================================================
def make_lung_template(nx: int = 32, ny: int = 32) -> np.ndarray:
    """
    Create a simple 32x32 synthetic lung-shaped template.
    """
    y, x = np.mgrid[0:ny, 0:nx]

    xn = (x - (nx - 1) / 2) / ((nx - 1) / 2)
    yn = (y - (ny - 1) / 2) / ((ny - 1) / 2)

    left = np.exp(-(((xn + 0.42) / 0.28) ** 2 + ((yn + 0.02) / 0.42) ** 2))
    right = np.exp(-(((xn - 0.42) / 0.28) ** 2 + ((yn + 0.02) / 0.42) ** 2))
    center_penalty = 0.35 * np.exp(-((xn / 0.16) ** 2 + ((yn + 0.02) / 0.7) ** 2))

    template = np.clip(left + right - center_penalty, 0, None)

    max_val = template.max()
    if max_val > 0:
        template = template / max_val

    return template.astype(np.float32)


def signal_to_pixel_impedance(signal: np.ndarray, seed: int = 123) -> np.ndarray:
    """
    Convert a 1D breathing waveform into a stack of 32x32 impedance frames.

    Returns
    -------
    pixel_impedance : ndarray
        Shape (n_frames, 32, 32), dtype float32
    """
    rng = np.random.default_rng(seed)
    template = make_lung_template()

    baseline = np.min(signal)
    dynamic = signal - baseline

    n_frames = len(signal)
    pixel_impedance = np.empty((n_frames, 32, 32), dtype=np.float32)

    for i, amp in enumerate(dynamic):
        frame = amp * template
        frame += rng.normal(0, 0.003 * max(amp, 1e-6), size=frame.shape)
        pixel_impedance[i] = frame.astype(np.float32)

    return pixel_impedance


# ==================================================
# 3) Phase / extrema detection for min_max_flag
# ==================================================
def detect_min_max_flags(signal: np.ndarray) -> np.ndarray:
    """
    Detect local minima and maxima for Draeger min_max_flag.

    Returns
    -------
    flags : ndarray of int32
        -1 for local minima
         1 for local maxima
         0 otherwise
    """
    n = len(signal)
    flags = np.zeros(n, dtype=np.int32)

    if n < 3:
        return flags

    d = np.diff(signal)

    for i in range(1, n - 1):
        prev_slope = d[i - 1]
        next_slope = d[i]

        if prev_slope > 0 and next_slope <= 0:
            flags[i] = 1
        elif prev_slope < 0 and next_slope >= 0:
            flags[i] = -1

    return flags


# ==================================================
# 4) Synthetic Medibus data
# ==================================================
def generate_medibus_data(
    time_seconds: np.ndarray,
    signal: np.ndarray,
    n_medibus_fields: int,
) -> np.ndarray:
    """
    Create synthetic Medibus channels.

    Only a few channels are populated with plausible values.
    Remaining channels are zero.
    """
    n_frames = len(time_seconds)
    medibus = np.zeros((n_medibus_fields, n_frames), dtype=np.float32)

    # Channel 0: airway pressure
    signal_range = np.ptp(signal)
    if signal_range == 0:
        scaled = np.zeros_like(signal)
    else:
        scaled = (signal - signal.min()) / signal_range
    medibus[0] = (8.0 + 6.0 * scaled).astype(np.float32)

    # Channel 1: flow
    ds = np.gradient(signal, time_seconds)
    medibus[1] = ds.astype(np.float32)

    # Channel 2: volume
    signal_std = np.std(signal)
    if signal_std == 0:
        medibus[2] = np.full_like(signal, 400.0, dtype=np.float32)
    else:
        medibus[2] = (400.0 + 150.0 * (signal - signal.mean()) / signal_std).astype(np.float32)

    # Respiratory rate field if present
    if n_medibus_fields > 36:
        medibus[36] = 14.0

    # Fraction inspired O2 field if present
    if n_medibus_fields > 44:
        medibus[44] = 21.0

    return medibus


# ==================================================
# 5) Event generation
# ==================================================
def generate_event_series(time_seconds: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """
    Create monotonically nondecreasing event markers and sparse event text.
    """
    n = len(time_seconds)
    event_markers = np.zeros(n, dtype=np.int32)
    event_texts = [""] * n

    events_to_insert = [
        (0.0, 1, "start"),
        (time_seconds[len(time_seconds) // 2], 2, "midpoint"),
    ]

    current_marker = 0
    j = 0

    for i, t in enumerate(time_seconds):
        while j < len(events_to_insert) and t >= events_to_insert[j][0]:
            current_marker = events_to_insert[j][1]
            event_texts[i] = events_to_insert[j][2]
            j += 1
        event_markers[i] = current_marker

    return event_markers, event_texts


# ==================================================
# 6) Build complete synthetic Draeger data
# ==================================================
def generate_synthetic_draeger_data(
    format_name: str = "original",
    duration: float = 60.0,
    fs: float = 20.0,
    seed: int = 42,
) -> SyntheticDraegerData:
    """
    Generate all arrays needed for a parser-compatible Draeger .bin file.
    """
    if format_name not in FORMAT_SPECS:
        raise ValueError(f"Unsupported format_name: {format_name}")

    spec = FORMAT_SPECS[format_name]
    n_medibus_fields = spec["n_medibus_fields"]

    time_seconds, signal = generate_realistic_eit_signal(
        duration=duration,
        fs=fs,
        seed=seed,
    )

    pixel_impedance = signal_to_pixel_impedance(signal, seed=seed + 1)
    medibus_data = generate_medibus_data(time_seconds, signal, n_medibus_fields=n_medibus_fields)
    min_max_flags = detect_min_max_flags(signal)
    event_markers, event_texts = generate_event_series(time_seconds)

    timing_errors = np.zeros(len(time_seconds), dtype=np.int32)
    unused_float32 = np.zeros(len(time_seconds), dtype=np.float32)

    return SyntheticDraegerData(
        format_name=format_name,
        frame_size=spec["frame_size"],
        sample_frequency=fs,
        time=time_seconds.astype(np.float64),
        pixel_impedance=pixel_impedance.astype(np.float32),
        medibus_data=medibus_data.astype(np.float32),
        min_max_flags=min_max_flags,
        event_markers=event_markers,
        event_texts=event_texts,
        timing_errors=timing_errors,
        unused_float32=unused_float32,
    )


# ==================================================
# 7) Exact frame packing based on _read_frame(...)
# ==================================================
def _encode_event_text(text: str, length: int = 30) -> bytes:
    """
    Encode text to exactly `length` bytes.
    Uses ASCII, truncates if needed, pads with NUL bytes.
    """
    raw = text.encode("ascii", errors="replace")[:length]
    return raw.ljust(length, b"\x00")


def pack_draeger_frame(
    time_seconds: float,
    pixel_frame: np.ndarray,
    min_max_flag: int,
    event_marker: int,
    event_text: str,
    timing_error: int,
    medibus_values: np.ndarray,
    unused_float32: float = 0.0,
) -> bytes:
    """
    Pack one Draeger frame exactly as expected by _read_frame(...).
    """
    pixel_frame = np.asarray(pixel_frame, dtype=np.float32)
    if pixel_frame.shape != (32, 32):
        raise ValueError("pixel_frame must have shape (32, 32)")

    medibus_values = np.asarray(medibus_values, dtype=np.float32).reshape(-1)

    # Loader does:
    # frame_time = reader.float64() * 24 * 60 * 60
    # Therefore we store time as fraction of day.
    time_fraction_of_day = float(time_seconds) / (24.0 * 60.0 * 60.0)

    parts = [
        struct.pack("<d", time_fraction_of_day),
        struct.pack("<f", float(unused_float32)),
        np.asarray(pixel_frame, dtype="<f4").reshape(-1, order="C").tobytes(),
        struct.pack("<i", int(min_max_flag)),
        struct.pack("<i", int(event_marker)),
        _encode_event_text(event_text, length=30),
        struct.pack("<i", int(timing_error)),
        np.asarray(medibus_values, dtype="<f4").tobytes(),
    ]
    return b"".join(parts)


# ==================================================
# 8) Save complete Draeger .bin file
# ==================================================
def save_draeger_bin(path: str | Path, data: SyntheticDraegerData) -> Path:
    """
    Save a complete Draeger-style binary file.
    """
    path = Path(path)

    n_frames = len(data.time)
    if data.pixel_impedance.shape != (n_frames, 32, 32):
        raise ValueError("pixel_impedance shape mismatch")
    if data.medibus_data.shape[1] != n_frames:
        raise ValueError("medibus_data shape mismatch")
    if len(data.min_max_flags) != n_frames:
        raise ValueError("min_max_flags length mismatch")
    if len(data.event_markers) != n_frames:
        raise ValueError("event_markers length mismatch")
    if len(data.event_texts) != n_frames:
        raise ValueError("event_texts length mismatch")
    if len(data.timing_errors) != n_frames:
        raise ValueError("timing_errors length mismatch")
    if len(data.unused_float32) != n_frames:
        raise ValueError("unused_float32 length mismatch")

    with path.open("wb") as f:
        for i in range(n_frames):
            frame_bytes = pack_draeger_frame(
                time_seconds=float(data.time[i]),
                pixel_frame=data.pixel_impedance[i],
                min_max_flag=int(data.min_max_flags[i]),
                event_marker=int(data.event_markers[i]),
                event_text=data.event_texts[i],
                timing_error=int(data.timing_errors[i]),
                medibus_values=data.medibus_data[:, i],
                unused_float32=float(data.unused_float32[i]),
            )

            if len(frame_bytes) != data.frame_size:
                raise ValueError(
                    f"Packed frame has {len(frame_bytes)} bytes, "
                    f"expected {data.frame_size}"
                )

            f.write(frame_bytes)

    return path


# ==================================================
# 9) Save preview plot
# ==================================================
def save_preview_plot(path: str | Path, data: SyntheticDraegerData) -> Path:
    """
    Save a simple preview plot of the global impedance signal.
    """
    path = Path(path)

    global_impedance = data.pixel_impedance.sum(axis=(1, 2))

    plt.figure(figsize=(12, 5.5))
    plt.plot(data.time, global_impedance, label="Global impedance")
    plt.xlabel("Time (s)")
    plt.ylabel("Impedance (a.u.)")
    plt.title(f"Synthetic Draeger-style EIT signal ({data.format_name})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    return path


# ==================================================
# 10) Main entry point
# ==================================================
def main() -> None:
    # Choose format:
    #   "original"     -> 52 Medibus fields, 4358 bytes/frame
    #   "pressure_pod" -> 58 Medibus fields, 4382 bytes/frame
    format_name = "original"

    # Generate synthetic data
    data = generate_synthetic_draeger_data(
        format_name=format_name,
        duration=60.0,
        fs=20.0,
        seed=42,
    )
    save_path = os.path.join(os.getcwd(), "test_data")
    # Save files
    bin_path = save_draeger_bin(os.path.join(save_path, "draeger_synthetic_draeger_20Hz.bin"), data)
    png_path = save_preview_plot(os.path.join(save_path,"draeger_synthetic_draeger_20Hz.png"), data)

    # Report
    print(f"Saved binary file: {bin_path}")
    print(f"Saved preview plot: {png_path}")
    print(f"Format: {data.format_name}")
    print(f"Frames: {len(data.time)}")
    print(f"Frame size: {data.frame_size} bytes")
    print(f"Expected file size: {len(data.time) * data.frame_size} bytes")
    print(f"Sample frequency: {data.sample_frequency} Hz")

    print("\nSuggested downstream usage:")
    print(
        'sequence = load_eit_data(\n'
        '    path="synthetic_draeger_20Hz.bin",\n'
        '    vendor="draeger",\n'
        '    sample_frequency=20.0,\n'
        '    label="synthetic_breathing",\n'
        '    name="Synthetic Draeger-style EIT",\n'
        '    description="Synthetic Draeger-style EIT frames generated from a realistic breathing waveform."\n'
        ')'
    )


if __name__ == "__main__":
    main()