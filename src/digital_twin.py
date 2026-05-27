"""Digital twin simulator for mechatronic rotor system.
Provides multi-channel telemetry: vibration, current, temperature, RPM.
Implements rotor dynamics, multi-fault physics, load/rpm variability, sensor drift and intermittent failures.
"""
from typing import Tuple, Dict
import numpy as np

# Physical constants (simplified)
J = 0.01  # rotor inertia kg m^2
B = 0.001  # damping
K_T = 0.1  # torque constant

FAULT_TYPES = [
    "healthy",
    "bearing_wear",
    "misalignment",
    "overload",
    "imbalance",
    "looseness",
    "multi_fault"
]


def _rotor_ode_step(omega: float, torque_net: float, dt: float) -> float:
    domega = (torque_net - B * omega) / J
    return omega + domega * dt


def _apply_sensor_failures(signal: np.ndarray, failure_rate: float, rng: np.random.Generator) -> np.ndarray:
    if failure_rate <= 0.0:
        return signal
    mask = rng.random(signal.shape) < failure_rate
    failed = signal.copy()
    failed[mask] = 0.0
    return failed


def simulate(duration_s: float,
             sample_rate: int,
             fault: str = "healthy",
             severity: str = "mild",
             speed_rpm: float = 1800.0,
             load_fraction: float = 0.5,
             ambient_temp: float = 25.0,
             rng_seed: int = 42) -> Tuple[np.ndarray, Dict]:
    """Generate a full synthetic run for the rotor system.

    Returns channels of shape (4, N): vibration, current, temperature, rpm.
    """
    rng = np.random.default_rng(rng_seed)
    N = int(duration_s * sample_rate)
    dt = 1.0 / sample_rate

    speed_rpm = max(300.0, float(speed_rpm))
    load_fraction = float(np.clip(load_fraction, 0.05, 0.95))
    severity_scale = {"mild": 0.7, "moderate": 1.0, "severe": 1.5}.get(severity, 1.0)
    severity_variation = float(np.clip(severity_scale + rng.normal(0.0, 0.12), 0.5, 1.6))

    torque_motor = K_T * 1.0
    torque_load_base = load_fraction * 0.6
    omega = speed_rpm * 2 * np.pi / 60.0 * 0.2

    rpm_trace = np.zeros(N, dtype=float)
    vibration = np.zeros(N, dtype=float)
    current = np.zeros(N, dtype=float)
    temperature = np.zeros(N, dtype=float)

    drift_rate = rng.normal(0.0, 1e-4, size=4)
    bias = rng.normal(0.0, [0.005, 0.02, 0.1, 0.0], size=4)
    failure_rate = 0.005

    base_temp = ambient_temp + 5.0 * load_fraction
    temp = base_temp
    rpm_target = speed_rpm
    active_faults = [fault]
    if fault == "multi_fault":
        active_faults = rng.choice(["bearing_wear", "misalignment", "overload", "imbalance", "looseness"], size=2, replace=False).tolist()
    severity_profile = {f: severity_variation * (0.75 + 0.25 * rng.random()) for f in active_faults}

    for i in range(N):
        t = i * dt
        rpm_fluct = 25.0 * np.sin(0.18 * t) + 12.0 * rng.standard_normal()
        rpm_cmd = rpm_target * (1.0 + 0.008 * np.sin(0.2 * t) + 0.008 * rng.standard_normal()) + rpm_fluct
        rpm_cmd = np.clip(rpm_cmd, 300.0, 2600.0)
        dyn_load = np.clip(load_fraction + 0.06 * np.sin(0.12 * t) + 0.03 * rng.standard_normal(), 0.05, 0.95)
        torque_load = torque_load_base * (1.0 + 0.07 * np.sin(0.13 * t) + 0.04 * rng.standard_normal()) * dyn_load

        overload_effect = 1.0 if "overload" in active_faults else 0.0
        torque_net = torque_motor - torque_load * (1.0 + 0.14 * overload_effect)
        omega = _rotor_ode_step(omega, torque_net, dt)
        rpm = omega * 60.0 / (2 * np.pi)
        rpm_trace[i] = rpm

        rot_freq = max(rpm, 1.0) / 60.0
        vib = 0.08 * np.sin(2 * np.pi * rot_freq * t)
        vib += 0.03 * np.sin(2 * np.pi * (rot_freq + 1.2) * t)

        if "bearing_wear" in active_faults:
            defect_freq = 10.0 * severity_profile["bearing_wear"]
            vib += 0.16 * severity_profile["bearing_wear"] * np.sign(np.sin(2 * np.pi * defect_freq * t)) * np.exp(-((i % 120) - 60)**2 / 400.0)

        if "misalignment" in active_faults:
            vib += 0.14 * severity_profile["misalignment"] * np.sin(2 * np.pi * 2.0 * rot_freq * t + 0.3)

        if "imbalance" in active_faults:
            vib += 0.09 * severity_profile["imbalance"] * np.sin(2 * np.pi * rot_freq * t + 0.5) * (1.0 + 0.2 * np.sin(0.05 * t))

        if "looseness" in active_faults:
            impulse = 0.22 * severity_profile["looseness"] * (rng.random() < 0.006)
            vib += impulse * np.exp(-0.5 * ((i % 50) / 10.0)**2)

        if overload_effect > 0:
            vib += 0.14 * overload_effect * rng.standard_normal()

        if fault == "healthy" and rng.random() < 0.18:
            vib += 0.04 * np.sin(2 * np.pi * (rot_freq * 1.5) * t + 0.2) + 0.02 * rng.standard_normal()
            vib += 0.02 * np.sign(np.sin(2 * np.pi * 8.0 * rot_freq * t)) * np.exp(-((i % 120) - 60)**2 / 500.0)

        cur = 0.35 + 0.38 * torque_load + 0.05 * np.sin(2 * np.pi * rot_freq * t)
        if overload_effect > 0:
            cur += 0.11 * severity_profile.get("overload", severity_variation)
        if "bearing_wear" in active_faults:
            cur += 0.035 * severity_profile["bearing_wear"]
        if "misalignment" in active_faults and rng.random() < 0.35:
            cur += 0.01 * severity_profile["misalignment"]

        temp += 0.0007 * (torque_load + overload_effect * 0.24) + 0.0004 * np.abs(rng.normal(0.0, 0.6))
        temp = ambient_temp + 3.0 * (dyn_load + 0.14 * overload_effect) + 0.22 * (rpm / 1000.0) + 0.05 * severity_profile.get(active_faults[0], severity_variation) + 0.1 * (temp - base_temp)

        vibration[i] = vib
        current[i] = cur
        temperature[i] = temp

    if rng.random() < 0.03:
        vibration = _apply_sensor_failures(vibration, failure_rate, rng)
    if rng.random() < 0.02:
        current = _apply_sensor_failures(current, failure_rate, rng)
    if rng.random() < 0.01:
        temperature = _apply_sensor_failures(temperature, failure_rate, rng)
    if rng.random() < 0.02:
        rpm_trace = _apply_sensor_failures(rpm_trace, failure_rate, rng)

    vibration = vibration + bias[0] + drift_rate[0] * np.arange(N)
    current = current + bias[1] + drift_rate[1] * np.arange(N)
    temperature = temperature + bias[2] + drift_rate[2] * np.arange(N)
    rpm_trace = rpm_trace + bias[3] + drift_rate[3] * np.arange(N)

    noise_scale = np.array([0.01, 0.02, 0.03, 0.2])[:, None]
    sensor_noise = rng.normal(loc=0.0, scale=noise_scale, size=(4, N))
    channels = np.vstack([vibration, current, temperature, rpm_trace])
    channel_gain = 1.0 + rng.normal(0.0, 0.02, size=(4, 1))
    channels = channels * channel_gain + sensor_noise
    if rng.random() < 0.12:
        spike_indices = rng.choice(N, size=max(1, int(0.01 * N)), replace=False)
        channels[0, spike_indices] += rng.normal(0.0, 0.25, size=spike_indices.shape)

    metadata = {
        "fault": fault,
        "severity": severity,
        "active_faults": active_faults,
        "speed_rpm": float(speed_rpm),
        "load_fraction": float(load_fraction),
        "ambient_temp": float(ambient_temp),
        "rng_seed": int(rng_seed)
    }
    return channels, metadata


if __name__ == "__main__":
    sig, meta = simulate(10.0, 2048, fault="bearing_wear", severity="moderate", speed_rpm=1500.0)
    print(sig.shape, meta)
