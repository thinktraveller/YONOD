"""ΔΔG <-> ee% conversion utilities for enantioselective reactions.

Physical relationship:
    ee (fraction) = tanh(ΔΔG / (2 * R * T))
    ee (percent)  = ee_fraction * 100

Where R = 0.001987 kcal/(mol·K) (gas constant in kcal units).

ΔΔG > 0 means the major enantiomer is favoured.
"""

from __future__ import annotations

import numpy as np


R_KCAL = 0.001987  # kcal / (mol · K)


def ddG_to_ee(ddG: float | np.ndarray, temperature_K: float | np.ndarray) -> np.ndarray:
    """Convert ΔΔG (kcal/mol) at temperature T (K) to ee%.

    Args:
        ddG:           scalar or array, kcal/mol
        temperature_K: scalar or array (broadcast-compatible with ddG), Kelvin

    Returns:
        ee%  in the range (-100, 100)
    """
    ddG_arr = np.asarray(ddG, dtype=np.float64)
    T_arr = np.asarray(temperature_K, dtype=np.float64)
    ee_frac = np.tanh(ddG_arr / (2.0 * R_KCAL * T_arr))
    return ee_frac * 100.0


def ee_mae(
    y_true_ddG: np.ndarray,
    y_pred_ddG: np.ndarray,
    temperatures_K: np.ndarray,
) -> float:
    """Mean Absolute Error in ee% space, converted from ΔΔG predictions.

    Useful as a chemically interpretable metric alongside R²/RMSE on ΔΔG.

    Args:
        y_true_ddG:    true ΔΔG values, kcal/mol
        y_pred_ddG:    predicted ΔΔG values, kcal/mol
        temperatures_K: reaction temperatures, Kelvin (same length)

    Returns:
        ee MAE in percentage points
    """
    ee_true = ddG_to_ee(y_true_ddG, temperatures_K)
    ee_pred = ddG_to_ee(y_pred_ddG, temperatures_K)
    return float(np.mean(np.abs(ee_true - ee_pred)))
