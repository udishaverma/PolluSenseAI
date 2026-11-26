from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

try:
    from .config import BACKEND_DIR, FORECAST_HOURS, POLLUTANTS
    from .forecasting import LiveForecaster
    from .aqi import calculate_aqi
except ImportError:
    from config import BACKEND_DIR, FORECAST_HOURS, POLLUTANTS
    from forecasting import LiveForecaster
    from aqi import calculate_aqi


DISPLAY_NAMES = {
    "co": "CO", "no2": "NO₂", "o3": "O₃",
    "pm10": "PM10", "pm25": "PM2.5", "so2": "SO₂"
}


def station_name(row):
    for col in ["station_name", "name", "station", "location_name"]:
        if col in row.index and pd.notna(row[col]):
            return str(row[col])
    return str(row["location_id"])


def build_response(results, forecaster):
    meta = forecaster.station_metadata.set_index("location_id")
    next_df = results["next_hour"]
    forecast = results["forecast_72h"]

    stations = {}
    for _, row in next_df.iterrows():
        sid = row["location_id"]
        vals = {p: float(row[p]) for p in POLLUTANTS}
        aqi = calculate_aqi(vals)
        m = meta.loc[sid]

        stations[sid] = {
            "station_id": sid,
            "station_name": station_name(m),
            "latitude": float(m["latitude"]),
            "longitude": float(m["longitude"]),
            "forecast_time_utc": pd.Timestamp(row["time"]).isoformat(),
            "pollutants": {
                p: {
                    "value": round(vals[p], 3),
                    "display_name": DISPLAY_NAMES[p]
                }
                for p in POLLUTANTS
            },
            "aqi": aqi["AQI"],
            "category": aqi["category"],
            "dominant_pollutant": aqi["dominant_pollutant"],
            "aqi_sub_indices": aqi["sub_indices"]
        }

    forecasts = {}
    for sid, df in forecast.groupby("location_id"):
        rows = []
        for _, row in df.sort_values("time").iterrows():
            vals = {p: float(row[p]) for p in POLLUTANTS}
            aqi = calculate_aqi(vals)
            rows.append({
                "time_utc": pd.Timestamp(row["time"]).isoformat(),
                "pollutants": {p: round(vals[p], 3) for p in POLLUTANTS},
                "aqi": aqi["AQI"],
                "category": aqi["category"],
                "dominant_pollutant": aqi["dominant_pollutant"]
            })
        forecasts[sid] = rows

    rankings = sorted(
        stations.values(),
        key=lambda x: (x["aqi"], x["station_name"].lower())
    )

    rankings = [
        {
            "rank": i,
            "station_id": s["station_id"],
            "station_name": s["station_name"],
            "aqi": s["aqi"],
            "category": s["category"],
            "dominant_pollutant": s["dominant_pollutant"]
        }
        for i, s in enumerate(rankings, 1)
    ]

    return {
        "application": {
            "name": "PolluSenseAI",
            "slogan": "Read the Air. Predict the Future."
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_data_time_utc": pd.Timestamp(
            results["live"]["time"].max()
        ).isoformat(),
        "next_forecast_time_utc": pd.Timestamp(
            results["next_hour_time"]
        ).isoformat(),
        "forecast_hours": FORECAST_HOURS,
        "station_count": len(stations),
        "pollutants": POLLUTANTS,
        "stations": stations,
        "forecasts": forecasts,
        "rankings": rankings
    }


def generate_live_response():
    forecaster = LiveForecaster()
    results = forecaster.run()
    response = build_response(results, forecaster)

    path = BACKEND_DIR / "live_response.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)

    return response


if __name__ == "__main__":
    print("Running PolluSenseAI backend...")
    response = generate_live_response()

    assert response["station_count"] == 37
    assert len(response["stations"]) == 37
    assert len(response["rankings"]) == 37
    assert len(response["forecasts"]) == 37
    assert all(
        len(v) == FORECAST_HOURS
        for v in response["forecasts"].values()
    )

    print(f"Stations: {response['station_count']}")
    print(f"Forecast hours: {response['forecast_hours']}")
    print(f"JSON: {BACKEND_DIR / 'live_response.json'}")
    print("✓ Response validation passed.")