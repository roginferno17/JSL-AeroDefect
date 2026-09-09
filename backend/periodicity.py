"""
JSL-AeroDefect: Roll-Mark Periodicity & Mechanical Root-Cause Diagnostic Engine
Applies Spatial Autocorrelation and Discrete Fourier Transform (DFT) to sequential
defect coordinates along the strip length to identify damaged work rolls and specific mill stands.
"""

import numpy as np
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    signal = None

# Standard Work Roll Geometries across Jindal Stainless Tandem & Sendzimir Mills
MILL_ROLL_REGISTRY = [
    {
        "stand_id": "Stand F1 (Roughing Pass)",
        "roll_type": "Work Roll (High Chrome Steel)",
        "diameter_mm": 420.0,
        "nominal_period_mm": 1319.5,
        "tolerance_mm": 35.0,
        "maintenance_action": "Schedule Stand F1 work-roll swap on next coil transfer. Check descaling header clearance.",
        "urgency": "MODERATE"
    },
    {
        "stand_id": "Stand F2 (Intermediate 1)",
        "roll_type": "Work Roll (High Speed Steel - HSS)",
        "diameter_mm": 380.0,
        "nominal_period_mm": 1193.8,
        "tolerance_mm": 30.0,
        "maintenance_action": "Inspect Stand F2 upper/lower roll surface for slag indentation. Execute roll grinder redress.",
        "urgency": "HIGH"
    },
    {
        "stand_id": "Stand F3 (Intermediate 2)",
        "roll_type": "Work Roll (Forged Steel)",
        "diameter_mm": 340.0,
        "nominal_period_mm": 1068.1,
        "tolerance_mm": 25.0,
        "maintenance_action": "Immediate inter-stand inspection. Verify roll coolant nozzles are unclogged to avoid thermal fatigue.",
        "urgency": "HIGH"
    },
    {
        "stand_id": "Stand F4 (Finishing)",
        "roll_type": "Tungsten Carbide Sleeve Roll",
        "diameter_mm": 300.0,
        "nominal_period_mm": 942.5,
        "tolerance_mm": 25.0,
        "maintenance_action": "CRITICAL: Surface marks will transmit directly to customer strip. Stop mill line; initiate auto roll change.",
        "urgency": "CRITICAL"
    },
    {
        "stand_id": "Stand F5 / Skin Pass Mill",
        "roll_type": "Textured / Mirror Polish Roll",
        "diameter_mm": 280.0,
        "nominal_period_mm": 879.6,
        "tolerance_mm": 20.0,
        "maintenance_action": "Skin pass roll blemish detected. Downgrade finish from 2B to No. 1 or schedule offline roll polishing.",
        "urgency": "CRITICAL"
    },
    {
        "stand_id": "Bridle / Deflector Roll #2",
        "roll_type": "Polyurethane / Chrome Plated Bridle",
        "diameter_mm": 250.0,
        "nominal_period_mm": 785.4,
        "tolerance_mm": 20.0,
        "maintenance_action": "Inspect tension bridle #2 for embedded slivers or guide contact wear.",
        "urgency": "MODERATE"
    }
]

class RollMarkPeriodicityAnalyzer:
    """
    Analyzes longitudinal coordinate series of detected roll marks
    using 1D Discrete Fourier Transform & Autocorrelation.
    """
    def __init__(self, forward_slip=0.035):
        # Forward slip coefficient (typically 3.0% - 4.0% in cold rolling)
        self.forward_slip = forward_slip

    def analyze_periodicity(self, longitudinal_positions_mm):
        """
        Input: Array of longitudinal positions (in millimeters) where roll marks were detected.
        Returns: Dominant repeating wavelength, estimated roll diameter, identified mill stand,
                 and FFT power spectrum for dashboard visualization.
        """
        if len(longitudinal_positions_mm) < 3:
            return {
                "periodic": False,
                "confidence": 0.0,
                "message": "Insufficient roll-mark sample count for spectral periodicity estimation (minimum 3 detections required).",
                "dominant_period_mm": None,
                "estimated_roll_diameter_mm": None,
                "identified_stand": None,
                "spectrum": []
            }

        positions = np.sort(np.array(longitudinal_positions_mm, dtype=np.float64))
        max_dist = positions[-1] - positions[0]
        
        # Calculate successive deltas
        deltas = np.diff(positions)
        
        # Construct continuous spatial impulse signal sampled at 10mm intervals
        sampling_step_mm = 10.0
        signal_length = int(np.ceil(max_dist / sampling_step_mm)) + 1
        signal_length = max(signal_length, 256)
        
        spatial_signal = np.zeros(signal_length)
        for p in positions:
            idx = int(round((p - positions[0]) / sampling_step_mm))
            if idx < signal_length:
                spatial_signal[idx] += 1.0

        # Gaussian smoothing to accommodate minor slip vibrations
        kernel = signal.windows.gaussian(7, std=1.5)
        smoothed = np.convolve(spatial_signal, kernel, mode='same')

        # 1D FFT Power Spectrum
        fft_vals = np.fft.rfft(smoothed - np.mean(smoothed))
        fft_freqs = np.fft.rfftfreq(len(smoothed), d=sampling_step_mm)  # cycles per mm
        power_spectrum = np.abs(fft_vals)**2

        # Convert frequencies to wavelengths (period in mm)
        # Avoid division by zero
        valid_mask = fft_freqs > (1.0 / 2500.0)  # Periods between 500mm and 2500mm
        filtered_freqs = fft_freqs[valid_mask]
        filtered_power = power_spectrum[valid_mask]

        if len(filtered_power) == 0:
            dominant_period = float(np.median(deltas))
            peak_power = 0.5
        else:
            peak_idx = np.argmax(filtered_power)
            dominant_freq = filtered_freqs[peak_idx]
            dominant_period = float(1.0 / dominant_freq) if dominant_freq > 0 else float(np.median(deltas))
            peak_power = float(filtered_power[peak_idx])

        # Roll diameter formula: D = lambda / (pi * (1 + s))
        estimated_diameter = dominant_period / (np.pi * (1.0 + self.forward_slip))
        
        # Match against JSL Roll Registry
        best_match = None
        min_error = float('inf')
        
        for roll in MILL_ROLL_REGISTRY:
            error = abs(roll["nominal_period_mm"] - dominant_period)
            if error < roll["tolerance_mm"] and error < min_error:
                min_error = error
                best_match = roll

        is_periodic = best_match is not None or (len(deltas) >= 3 and np.std(deltas) / (np.mean(deltas) + 1e-5) < 0.15)
        match_confidence = round(max(0.0, min(1.0, 1.0 - (min_error / 50.0))) * 100.0, 1) if best_match else 0.0

        # Generate lightweight spectrum points for UI charts
        spectrum_chart = []
        step = max(1, len(filtered_freqs) // 30)
        for i in range(0, len(filtered_freqs), step):
            wavelength = 1.0 / filtered_freqs[i]
            if 600 <= wavelength <= 1600:
                spectrum_chart.append({
                    "wavelength_mm": round(wavelength, 1),
                    "power": round(float(filtered_power[i] / (np.max(filtered_power) + 1e-6)), 4)
                })

        return {
            "periodic": bool(is_periodic),
            "confidence_pct": match_confidence,
            "dominant_period_mm": round(dominant_period, 1),
            "estimated_roll_diameter_mm": round(estimated_diameter, 1),
            "identified_stand": best_match["stand_id"] if best_match else "Unknown Stand / Non-Periodic Mark",
            "roll_type": best_match["roll_type"] if best_match else "N/A",
            "urgency": best_match["urgency"] if best_match else "LOW",
            "maintenance_action": best_match["maintenance_action"] if best_match else "Continue monitoring defect progression.",
            "spectrum": spectrum_chart,
            "sample_detections_count": len(positions)
        }
