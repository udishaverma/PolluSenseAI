from pathlib import Path
import inspect
import sys
import numpy as np

try:
    from .config import PROJECT_ROOT, POLLUTANTS
    from .forecasting import LiveForecaster
except ImportError:
    from config import PROJECT_ROOT, POLLUTANTS
    from forecasting import LiveForecaster


XAI_DIR = Path(PROJECT_ROOT) / "xai"
sys.path.insert(0, str(XAI_DIR))

from pollusense_xai import explain_prediction


FEATURE_NAMES = {
    "era5_t2m_c": "Temperature",
    "era5_d2m_c": "Dew-point temperature",
    "era5_sp_hpa": "Surface pressure",
    "era5_tp_mm": "Precipitation",
    "era5_wind_speed": "Wind speed",
    "era5_wind_dir": "Wind direction",
    "era5_blh": "Boundary-layer height",

    "hour": "Hour of day",
    "month": "Month",
    "dayofweek": "Day of week",

    "hour_sin": "Time-of-day sine component",
    "hour_cos": "Time-of-day cosine component",

    "dow_sin": "Day-of-week sine component",
    "dow_cos": "Day-of-week cosine component",

    "pm25_history": "Recent PM2.5 levels",
    "pm10_history": "Recent PM10 levels",
    "no2_history": "Recent NO₂ levels",
    "o3_history": "Recent O₃ levels",
    "so2_history": "Recent SO₂ levels",
    "co_history": "Recent CO levels",

    "pm25_lag1": "Previous-hour PM2.5 level",
    "pm10_lag1": "Previous-hour PM10 level",
    "no2_lag1": "Previous-hour NO₂ level",
    "o3_lag1": "Previous-hour O₃ level",
    "so2_lag1": "Previous-hour SO₂ level",
    "co_lag1": "Previous-hour CO level",

    "pm25_lag24": "PM2.5 level 24 hours earlier",
    "pm10_lag24": "PM10 level 24 hours earlier",
    "no2_lag24": "NO₂ level 24 hours earlier",
    "o3_lag24": "O₃ level 24 hours earlier",
    "so2_lag24": "SO₂ level 24 hours earlier",
    "co_lag24": "CO level 24 hours earlier",

    "pm25_roll24_mean": "24-hour average PM2.5",
    "pm10_roll24_mean": "24-hour average PM10",
    "no2_roll24_mean": "24-hour average NO₂",
    "o3_roll24_mean": "24-hour average O₃",
    "so2_roll24_mean": "24-hour average SO₂",
    "co_roll24_mean": "24-hour average CO",
}


def pretty_feature(name):
    return FEATURE_NAMES.get(
        name,
        name.replace("_", " ").title(),
    )


def explain_station(
    forecaster,
    live,
    station="ST01",
    pollutant="pm25",
):
    if station not in forecaster.stations:
        raise ValueError(f"Unknown station: {station}")

    if pollutant not in POLLUTANTS:
        raise ValueError(f"Unknown pollutant: {pollutant}")

    live = forecaster.engineer_features(live)

    # Use the latest timestamp containing complete
    # pollutant observations for every station.
    complete_hours = (
        live.dropna(subset=forecaster.targets)
        .groupby("time")["location_id"]
        .nunique()
    )

    complete_hours = complete_hours[
        complete_hours == len(forecaster.stations)
    ]

    if complete_hours.empty:
        raise ValueError(
            "No complete pollutant hour is available for XAI."
        )

    latest = complete_hours.index.max()

    X = forecaster.build_model_tensor(
        live,
        latest,
    )

    import torch

    xseq = torch.tensor(
        X[None],
        dtype=torch.float32,
        device=forecaster.device,
    )

    current = (
        live[
            live["time"] == latest
        ]
        .set_index("location_id")
    )

    if station not in current.index:
        raise ValueError(
            f"{station} is missing at XAI timestamp {latest}."
        )

    last_value = float(
        current.loc[station, pollutant]
    )

    args = {
        "model": forecaster.model,
        "xseq": xseq,
        "edge_index": forecaster.edge_index,
        "edge_attr": forecaster.edge_attr,
        "stations": forecaster.stations,
        "features": forecaster.features,
        "targets": forecaster.targets,
        "station": station,
        "pollutant": pollutant,
        "res_mean": forecaster.res_mean,
        "res_std": forecaster.res_std,
        "last_value": last_value,
        "background_hours": 8,
        "top_k": 10,
    }

    params = inspect.signature(
        explain_prediction
    ).parameters

    args = {
        k: v
        for k, v in args.items()
        if k in params
    }

    return explain_prediction(**args)


def natural_language_summary(
    xai,
    station_names=None,
):
    station_names = station_names or {}

    station_id = xai.get(
        "station",
        "Unknown station",
    )

    station_name = station_names.get(
        station_id,
        station_id,
    )

    pollutant = xai.get(
        "pollutant",
        "",
    )

    prediction = float(
        xai.get("prediction", 0.0)
    )

    unit = xai.get(
        "unit",
        "",
    )

    positive = [
        pretty_feature(feature)
        for feature in xai.get(
            "top_positive_features",
            [],
        )[:3]
    ]

    negative = [
        pretty_feature(feature)
        for feature in xai.get(
            "top_negative_features",
            [],
        )[:3]
    ]

    spatial = []

    for item in xai.get(
        "spatial_attention",
        [],
    )[:3]:
        spatial_id = item.get(
            "station",
            "Unknown station",
        )

        spatial.append(
            station_names.get(
                spatial_id,
                spatial_id,
            )
        )

    positive_text = (
        ", ".join(positive)
        if positive
        else "none identified"
    )

    negative_text = (
        ", ".join(negative)
        if negative
        else "none identified"
    )

    spatial_text = (
        ", ".join(spatial)
        if spatial
        else "none identified"
    )

    return (
        f"For {station_name}, the GAT-LSTM predicted "
        f"{prediction:.2f} {unit} for "
        f"{pollutant.upper()}. "
        f"The strongest positive model influences were "
        f"{positive_text}, while the strongest negative "
        f"influences were {negative_text}. "
        f"The highest spatial attention was assigned to "
        f"{spatial_text}."
    )


if __name__ == "__main__":
    print("PolluSenseAI XAI")
    print("=" * 45)

    forecaster = LiveForecaster()
    results = forecaster.run()

    station_names = dict(
        zip(
            forecaster.station_metadata["location_id"],
            forecaster.station_metadata["station_name"],
        )
    )

    xai = explain_station(
        forecaster,
        results["live"],
        station="ST01",
        pollutant="pm25",
    )

    xai["explanation"] = natural_language_summary(
        xai,
        station_names,
    )

    print("Station:", station_names.get(
        xai["station"],
        xai["station"],
    ))

    print(
        "Pollutant:",
        xai["pollutant"].upper(),
    )

    print(
        "Prediction:",
        round(xai["prediction"], 2),
        xai["unit"],
    )

    print(
        "\nTop spatial attention:",
        [
            station_names.get(
                item["station"],
                item["station"],
            )
            for item in xai["spatial_attention"][:3]
        ],
    )

    print(
        "Top positive SHAP:",
        [
            pretty_feature(feature)
            for feature
            in xai["top_positive_features"][:3]
        ],
    )

    print(
        "Top negative SHAP:",
        [
            pretty_feature(feature)
            for feature
            in xai["top_negative_features"][:3]
        ],
    )

    print(
        "\nExplanation:",
        xai["explanation"],
    )

    print("\n✓ XAI validation passed.")