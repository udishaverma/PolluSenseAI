# ============================================================
# backend/aqi.py
# PolluSenseAI - CPCB AQI Calculation
# ============================================================

from typing import Dict, List, Tuple

import math


# ============================================================
# CPCB AQI Breakpoints
# ============================================================
#
# Format:
# (pollutant concentration lower,
#  pollutant concentration upper,
#  AQI lower,
#  AQI upper)
#
# These are the exact breakpoints used in Notebook 10.
# ============================================================

BP: Dict[str, List[Tuple[float, float, int, int]]] = {
    "co": [
        (0, 1, 0, 50),
        (1, 2, 51, 100),
        (2, 10, 101, 200),
        (10, 17, 201, 300),
        (17, 34, 301, 400),
        (34, math.inf, 401, 500),
    ],

    "no2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 180, 101, 200),
        (180, 280, 201, 300),
        (280, 400, 301, 400),
        (400, math.inf, 401, 500),
    ],

    "o3": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 168, 101, 200),
        (168, 208, 201, 300),
        (208, 748, 301, 400),
        (748, math.inf, 401, 500),
    ],

    "pm10": [
        (0, 50, 0, 50),
        (50, 100, 51, 100),
        (100, 250, 101, 200),
        (250, 350, 201, 300),
        (350, 430, 301, 400),
        (430, math.inf, 401, 500),
    ],

    "pm25": [
        (0, 30, 0, 50),
        (30, 60, 51, 100),
        (60, 90, 101, 200),
        (90, 120, 201, 300),
        (120, 250, 301, 400),
        (250, math.inf, 401, 500),
    ],

    "so2": [
        (0, 40, 0, 50),
        (40, 80, 51, 100),
        (80, 380, 101, 200),
        (380, 800, 201, 300),
        (800, 1600, 301, 400),
        (1600, math.inf, 401, 500),
    ],
}


# ============================================================
# AQI Category
# ============================================================

def aqi_category(aqi: float) -> str:
    """
    Convert AQI value into CPCB AQI category.
    """

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Satisfactory"

    if aqi <= 200:
        return "Moderate"

    if aqi <= 300:
        return "Poor"

    if aqi <= 400:
        return "Very Poor"

    return "Severe"


# ============================================================
# Single Pollutant Sub-Index
# ============================================================

def cpcb_aqi(value: float, pollutant: str) -> float:
    """
    Calculate the CPCB pollutant sub-index.

    Parameters
    ----------
    value:
        Pollutant concentration.

    pollutant:
        One of:
        co, no2, o3, pm10, pm25, so2

    Returns
    -------
    float
        Pollutant-specific AQI sub-index.
    """

    pollutant = pollutant.lower()

    if pollutant not in BP:
        raise ValueError(
            f"Unsupported pollutant '{pollutant}'. "
            f"Supported pollutants: {list(BP.keys())}"
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"Invalid concentration for {pollutant}: {value}"
        )

    # Negative concentration is not physically meaningful.
    # Treat it as zero for safe inference.
    value = max(value, 0.0)

    for lo, hi, aqi_lo, aqi_hi in BP[pollutant]:

        if lo <= value <= hi:
            return (
                aqi_lo
                + (aqi_hi - aqi_lo)
                * (value - lo)
                / (hi - lo)
            )

    # Values above the final breakpoint are capped at 500.
    return 500.0


# ============================================================
# Calculate AQI from all six predicted pollutants
# ============================================================

def calculate_aqi(
    pollutant_values: Dict[str, float],
) -> Dict[str, object]:
    """
    Calculate CPCB AQI from predicted pollutant concentrations.

    Parameters
    ----------
    pollutant_values:
        Dictionary containing:
            co
            no2
            o3
            pm10
            pm25
            so2

    Returns
    -------
    dict
        {
            "AQI": int,
            "category": str,
            "dominant_pollutant": str,
            "sub_indices": dict
        }
    """

    required = [
        "co",
        "no2",
        "o3",
        "pm10",
        "pm25",
        "so2",
    ]

    missing = [
        pollutant
        for pollutant in required
        if pollutant not in pollutant_values
    ]

    if missing:
        raise ValueError(
            f"Missing pollutant values: {missing}"
        )

    sub_indices = {
        pollutant: cpcb_aqi(
            pollutant_values[pollutant],
            pollutant,
        )
        for pollutant in required
    }

    # CPCB AQI is determined by the maximum
    # pollutant-specific sub-index.
    dominant_pollutant = max(
        sub_indices,
        key=sub_indices.get,
    )

    aqi = sub_indices[dominant_pollutant]

    return {
        "AQI": round(aqi),
        "category": aqi_category(aqi),
        "dominant_pollutant": dominant_pollutant,
        "sub_indices": {
            pollutant: round(value, 2)
            for pollutant, value in sub_indices.items()
        },
    }


# ============================================================
# Convenience function for a prediction array
# ============================================================

def calculate_aqi_from_prediction(
    prediction,
    target_order=None,
) -> Dict[str, object]:
    """
    Calculate AQI directly from a six-element prediction array/list.

    Default target order:
        CO, NO2, O3, PM10, PM2.5, SO2

    Parameters
    ----------
    prediction:
        Array-like with six pollutant predictions.

    target_order:
        Optional custom pollutant order.

    Returns
    -------
    dict
        Same structure as calculate_aqi().
    """

    if target_order is None:
        target_order = [
            "co",
            "no2",
            "o3",
            "pm10",
            "pm25",
            "so2",
        ]

    if len(prediction) != len(target_order):
        raise ValueError(
            f"Expected {len(target_order)} predictions, "
            f"got {len(prediction)}."
        )

    values = {
        pollutant: float(value)
        for pollutant, value in zip(
            target_order,
            prediction,
        )
    }

    return calculate_aqi(values)


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    print("PolluSenseAI CPCB AQI Module")
    print("=" * 45)

    # Example prediction, using the same six pollutants
    # produced by the trained model.
    example = {
        "co": 0.8,
        "no2": 42.0,
        "o3": 55.0,
        "pm10": 62.0,
        "pm25": 24.0,
        "so2": 18.0,
    }

    result = calculate_aqi(example)

    print("Predicted pollutants:")
    for pollutant, value in example.items():
        print(f"  {pollutant:5s}: {value}")

    print("\nPollutant sub-indices:")
    for pollutant, value in result["sub_indices"].items():
        print(f"  {pollutant:5s}: {value}")

    print("\nFinal AQI:", result["AQI"])
    print("Category:", result["category"])
    print("Dominant pollutant:", result["dominant_pollutant"])

    # Basic validation
    assert 0 <= result["AQI"] <= 500
    assert result["category"] in {
        "Good",
        "Satisfactory",
        "Moderate",
        "Poor",
        "Very Poor",
        "Severe",
    }
    assert result["dominant_pollutant"] in BP

    print("\n✓ AQI module validation passed.")