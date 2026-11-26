# ============================================================
# backend/forecasting.py
# PolluSenseAI - Live Forecasting Engine
# ============================================================

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
import torch

try:
    from .config import (
        OPENWEATHER_API_KEY,
        OPENWEATHER_BASE_URL,
        OPENWEATHER_HISTORY_URL,
        OPENMETEO_FORECAST_URL,
        NUM_STATIONS,
        NUM_INPUT_CHANNELS,
        POLLUTANTS,
        SEQUENCE_LENGTH,
        FORECAST_HOURS,
    )
    from .model import ModelBundle
except ImportError:
    from config import (
        OPENWEATHER_API_KEY,
        OPENWEATHER_BASE_URL,
        OPENWEATHER_HISTORY_URL,
        OPENMETEO_FORECAST_URL,
        NUM_STATIONS,
        NUM_INPUT_CHANNELS,
        POLLUTANTS,
        SEQUENCE_LENGTH,
        FORECAST_HOURS,
    )
    from model import ModelBundle


# ============================================================
# API / weather configuration
# ============================================================

OPENWEATHER_TIMEOUT = 30
OPENMETEO_TIMEOUT = 60

WEATHER_COLUMNS = [
    "location_id",
    "time",
    "latitude",
    "longitude",
    "era5_t2m_c",
    "era5_d2m_c",
    "era5_sp_hpa",
    "era5_tp_mm",
    "era5_wind_speed",
    "era5_wind_dir",
    "era5_blh",
]


# ============================================================
# Live Forecasting Engine
# ============================================================

class LiveForecaster:
    """
    Complete live forecasting pipeline.

    Pipeline:
        OpenWeather pollutant history
            +
        Open-Meteo weather history/forecast
            ↓
        feature engineering
            ↓
        24-hour × 37-station × 46-channel input
            ↓
        trained GAT-LSTM
            ↓
        recursive 72-hour pollutant forecast
    """

    def __init__(self, model_bundle: ModelBundle | None = None):
        if not OPENWEATHER_API_KEY:
            raise RuntimeError(
                "OPENWEATHER_API_KEY was not found. "
                "Make sure your project-root .env file contains it."
            )

        # Load the trained model once.
        # Passing an existing bundle avoids reloading it.
        self.bundle = model_bundle or ModelBundle()

        self.model = self.bundle.model
        self.device = self.bundle.device

        self.features = list(self.bundle.features)
        self.targets = list(self.bundle.targets)
        self.stations = list(self.bundle.stations)

        self.seq_len = int(self.bundle.seq_len)

        self.fill_values = np.asarray(
            self.bundle.fill_values,
            dtype=np.float32,
        )

        self.target_mean = np.asarray(
            self.bundle.target_mean,
            dtype=np.float32,
        )

        self.target_std = np.asarray(
            self.bundle.target_std,
            dtype=np.float32,
        )

        self.res_mean = np.asarray(
            self.bundle.res_mean,
            dtype=np.float32,
        )

        self.res_std = np.asarray(
            self.bundle.res_std,
            dtype=np.float32,
        )

        self.scaler = self.bundle.feature_scaler

        # Station metadata is already ordered according to
        # the model's station list by ModelBundle.
        self.station_metadata = (
            self.bundle.station_metadata.copy()
        )

        # Graph artifacts are already loaded by ModelBundle.
        self.edge_index = self.bundle.edge_index.to(
            self.device
        )
        self.edge_attr = self.bundle.edge_attr.to(
            self.device
        )

        self._validate_setup()

    # ========================================================
    # Validation
    # ========================================================

    def _validate_setup(self) -> None:
        """Validate dimensions and model configuration."""

        if len(self.stations) != NUM_STATIONS:
            raise ValueError(
                f"Expected {NUM_STATIONS} stations, "
                f"found {len(self.stations)}."
            )

        if self.seq_len != SEQUENCE_LENGTH:
            raise ValueError(
                f"Expected sequence length {SEQUENCE_LENGTH}, "
                f"found {self.seq_len}."
            )

        if len(self.features) != 34:
            raise ValueError(
                f"Expected 34 engineered features, "
                f"found {len(self.features)}."
            )

        if len(self.targets) != len(POLLUTANTS):
            raise ValueError(
                f"Expected {len(POLLUTANTS)} targets, "
                f"found {len(self.targets)}."
            )

        if NUM_INPUT_CHANNELS != (
            len(self.features) + len(self.targets) + len(self.targets)
        ):
            raise ValueError(
                "Configured 46-channel input does not match "
                "34 base + 6 history + 6 availability channels."
            )

        if self.edge_index.shape[1] != 169:
            raise ValueError(
                f"Expected 169 graph edges, "
                f"found {self.edge_index.shape[1]}."
            )

    # ========================================================
    # OpenWeather pollutant history
    # ========================================================

    def fetch_openweather_history(
        self,
        hours: int = 48,
    ) -> pd.DataFrame:
        """
        Fetch historical pollutant observations from OpenWeather.

        Returns:
            DataFrame containing one row per station/time.
        """

        if hours <= 0:
            raise ValueError("hours must be positive.")

        end = int(
            datetime.now(timezone.utc).timestamp()
        )

        start = int(
            (
                datetime.now(timezone.utc)
                - timedelta(hours=hours)
            ).timestamp()
        )

        rows = []

        for _, station in self.station_metadata.iterrows():

            response = requests.get(
                OPENWEATHER_HISTORY_URL,
                params={
                    "lat": station.latitude,
                    "lon": station.longitude,
                    "start": start,
                    "end": end,
                    "appid": OPENWEATHER_API_KEY,
                },
                timeout=OPENWEATHER_TIMEOUT,
            )

            response.raise_for_status()

            for item in response.json().get("list", []):
                rows.append(
                    {
                        "location_id": station.location_id,
                        "time": pd.to_datetime(
                            item["dt"],
                            unit="s",
                            utc=True,
                        ),
                        **item["components"],
                    }
                )

        if not rows:
            raise RuntimeError(
                "OpenWeather returned no historical pollutant data."
            )

        df = pd.DataFrame(rows)

        df = df.rename(
            columns={"pm2_5": "pm25"}
        )

        # Notebook 10 converts OpenWeather CO to the
        # model's CO scale.
        df["co"] = df["co"] / 1000.0

        return df

    # ========================================================
    # OpenWeather current observation
    # ========================================================

    def fetch_openweather_current(
        self,
    ) -> pd.DataFrame:
        """
        Fetch the latest OpenWeather observation for all stations.
        """

        rows = []

        for _, station in self.station_metadata.iterrows():

            response = requests.get(
                OPENWEATHER_BASE_URL,
                params={
                    "lat": station.latitude,
                    "lon": station.longitude,
                    "appid": OPENWEATHER_API_KEY,
                },
                timeout=OPENWEATHER_TIMEOUT,
            )

            response.raise_for_status()

            payload = response.json()

            if not payload.get("list"):
                raise RuntimeError(
                    f"OpenWeather returned no current data "
                    f"for {station.location_id}."
                )

            item = payload["list"][0]

            rows.append(
                {
                    "location_id": station.location_id,
                    "time": pd.to_datetime(
                        item["dt"],
                        unit="s",
                        utc=True,
                    ),
                    **item["components"],
                }
            )

        df = pd.DataFrame(rows)

        df = df.rename(
            columns={"pm2_5": "pm25"}
        )

        # Match Notebook 10 scaling.
        df["co"] = df["co"] / 1000.0

        return df

    # ========================================================
    # Open-Meteo weather
    # ========================================================

    def fetch_weather(
        self,
        past_hours: int = 48,
        forecast_hours: int = 73,
    ) -> pd.DataFrame:
        """
        Fetch historical + forecast meteorological variables
        from Open-Meteo.
        """

        latitudes = ",".join(
            self.station_metadata.latitude.astype(str)
        )

        longitudes = ",".join(
            self.station_metadata.longitude.astype(str)
        )

        params = {
            "latitude": latitudes,
            "longitude": longitudes,
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "dew_point_2m",
                    "surface_pressure",
                    "precipitation",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "boundary_layer_height",
                ]
            ),
            "past_hours": past_hours,
            "forecast_hours": forecast_hours,
            "timezone": "UTC",
        }

        response = requests.get(
            OPENMETEO_FORECAST_URL,
            params=params,
            timeout=OPENMETEO_TIMEOUT,
        )

        response.raise_for_status()

        data = response.json()

        # Open-Meteo returns a list when multiple coordinates
        # are supplied.
        if not isinstance(data, list):
            raise RuntimeError(
                "Unexpected Open-Meteo response format."
            )

        rows = []

        for index, station_id in enumerate(self.stations):

            hourly = data[index].get("hourly")

            if hourly is None:
                raise RuntimeError(
                    f"Open-Meteo returned no hourly data "
                    f"for {station_id}."
                )

            station_df = pd.DataFrame(hourly)

            station_df["location_id"] = station_id

            rows.append(station_df)

        weather = pd.concat(
            rows,
            ignore_index=True,
        )

        weather["time"] = pd.to_datetime(
            weather["time"],
            utc=True,
        )

        weather = weather.rename(
            columns={
                "temperature_2m": "era5_t2m_c",
                "dew_point_2m": "era5_d2m_c",
                "surface_pressure": "era5_sp_hpa",
                "precipitation": "era5_tp_mm",
                "wind_speed_10m": "era5_wind_speed",
                "wind_direction_10m": "era5_wind_dir",
                "boundary_layer_height": "era5_blh",
            }
        )

        return weather

    # ========================================================
    # Build live pollutant + weather dataset
    # ========================================================

    def build_live_dataset(
        self,
        pollutant_history: pd.DataFrame,
        current_pollution: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge OpenWeather pollutant data and Open-Meteo weather
        exactly as required by the model.
        """

        hist = pollutant_history.copy()
        current = current_pollution.copy()

        # Notebook 10 floors the current timestamp to the hour.
        current["time"] = current["time"].dt.floor("h")

        # Prevent duplication of the current hour.
        hist = hist[
            hist["time"] < current["time"].max()
        ]

        pollution = pd.concat(
            [
                hist,
                current,
            ],
            ignore_index=True,
        )

        pollution = pollution.drop_duplicates(
            ["location_id", "time"],
            keep="last",
        )

        live = (
            pollution
            .merge(
                weather,
                on=["location_id", "time"],
                how="inner",
            )
            .merge(
                self.station_metadata[
                    [
                        "location_id",
                        "latitude",
                        "longitude",
                    ]
                ],
                on="location_id",
                how="left",
            )
            .sort_values(
                ["location_id", "time"]
            )
            .reset_index(drop=True)
        )

        if live.empty:
            raise RuntimeError(
                "Live pollutant/weather merge produced no rows."
            )

        return live

    # ========================================================
    # Feature engineering
    # ========================================================

    def engineer_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate the exact 34 engineered features used by
        the trained model.
        """

        result = df.copy()

        result = result.sort_values(
            ["location_id", "time"]
        ).reset_index(drop=True)

        # ----------------------------------------------------
        # Calendar features
        # ----------------------------------------------------

        result["hour"] = result["time"].dt.hour
        result["month"] = result["time"].dt.month
        result["dayofweek"] = (
            result["time"].dt.dayofweek
        )

        result["hour_sin"] = np.sin(
            2 * np.pi * result["hour"] / 24
        )

        result["hour_cos"] = np.cos(
            2 * np.pi * result["hour"] / 24
        )

        result["dow_sin"] = np.sin(
            2 * np.pi * result["dayofweek"] / 7
        )

        result["dow_cos"] = np.cos(
            2 * np.pi * result["dayofweek"] / 7
        )

        # ----------------------------------------------------
        # Pollutant temporal features
        # ----------------------------------------------------

        for pollutant in self.targets:

            group = result.groupby(
                "location_id"
            )[pollutant]

            result[
                f"{pollutant}_lag1"
            ] = group.shift(1)

            result[
                f"{pollutant}_lag24"
            ] = group.shift(24)

            result[
                f"{pollutant}_roll24_mean"
            ] = (
                group
                .shift(1)
                .rolling(
                    24,
                    min_periods=1,
                )
                .mean()
                .reset_index(
                    level=0,
                    drop=True,
                )
            )

        return result

    # ========================================================
    # Build 46-channel model tensor
    # ========================================================

    def build_model_tensor(
        self,
        df: pd.DataFrame,
        end_time: pd.Timestamp,
    ) -> np.ndarray:
        """
        Construct:
            [24, 37, 46]

        channels:
            34 engineered/scaled features
            6 pollutant history channels
            6 availability channels
        """

        window_times = pd.date_range(
            end_time
            - pd.Timedelta(
                hours=self.seq_len - 1
            ),
            end_time,
            freq="h",
        )

        grid = pd.MultiIndex.from_product(
            [
                window_times,
                self.stations,
            ],
            names=[
                "time",
                "location_id",
            ],
        )

        indexed = (
            df
            .set_index(
                [
                    "time",
                    "location_id",
                ]
            )
            .reindex(grid)
        )

        # ----------------------------------------------------
        # 34 base features
        # ----------------------------------------------------

        base = indexed[
            self.features
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        fill_series = pd.Series(
            self.fill_values,
            index=self.features,
        )

        base = base.fillna(
            fill_series
        )

        base_scaled = (
            self.scaler
            .transform(
                base.to_numpy(
                    dtype=np.float32
                )
            )
            .astype(np.float32)
        )

        # ----------------------------------------------------
        # Pollutant history channels
        # ----------------------------------------------------

        history = indexed[
            self.targets
        ]

        availability = (
            history.notna()
            .to_numpy(
                dtype=np.float32
            )
        )

        history_scaled = (
            history.to_numpy(
                dtype=np.float32
            )
            - self.target_mean
        ) / self.target_std

        history_scaled = np.nan_to_num(
            history_scaled,
            nan=0.0,
        ).astype(np.float32)

        # ----------------------------------------------------
        # Final 46 channels
        # ----------------------------------------------------

        combined = np.concatenate(
            [
                base_scaled,
                history_scaled,
                availability,
            ],
            axis=1,
        )

        X = combined.reshape(
            self.seq_len,
            NUM_STATIONS,
            NUM_INPUT_CHANNELS,
        )

        if np.isnan(X).any():
            raise ValueError(
                "Model tensor contains NaN values."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Model tensor contains non-finite values."
            )

        return X.astype(np.float32)

    # ========================================================
    # Raw model prediction
    # ========================================================

    def predict_raw(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Run the trained model.

        Input:
            [24, 37, 46]

        Output:
            [37, 6]

        The output is still in the model's residual-scaled
        space. Use reconstruct_prediction() afterward.
        """

        expected_shape = (
            self.seq_len,
            NUM_STATIONS,
            NUM_INPUT_CHANNELS,
        )

        if X.shape != expected_shape:
            raise ValueError(
                f"Expected tensor shape "
                f"{expected_shape}, "
                f"got {X.shape}."
            )

        tensor = torch.tensor(
            X[None],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            raw = self.model(
                tensor,
                self.edge_index,
                self.edge_attr,
            )[0]

        return raw.cpu().numpy()

    # ========================================================
    # Reconstruct pollutant predictions
    # ========================================================

    def reconstruct_prediction(
        self,
        raw: np.ndarray,
        previous_pollutants: pd.DataFrame,
    ) -> np.ndarray:
        """
        Convert model residual output into pollutant
        concentration predictions.

        Notebook 10 reconstruction:

            prediction =
                raw * residual_std
                + residual_mean
                + previous concentration
        """

        previous = (
            previous_pollutants
            .set_index("location_id")
            [self.targets]
            .reindex(self.stations)
        )

        if previous.isna().any().any():
            raise ValueError(
                "Previous pollutant values contain missing "
                "station values."
            )

        previous_values = previous.to_numpy(
            dtype=np.float32
        )

        output = np.zeros(
            (
                NUM_STATIONS,
                len(self.targets),
            ),
            dtype=np.float32,
        )

        for k, pollutant in enumerate(
            self.targets
        ):
            output[:, k] = (
                raw[:, k]
                * self.res_std[k]
                + self.res_mean[k]
                + previous_values[:, k]
            )

        # Pollutant concentrations cannot be negative.
        output = np.clip(
            output,
            0,
            None,
        )

        return output

    # ========================================================
    # Complete current / next-hour forecast
    # ========================================================

    def forecast_next_hour(
        self,
        live: pd.DataFrame,
    ) -> Tuple[
        pd.Timestamp,
        pd.DataFrame,
    ]:
        """
        Predict the next hour for all 37 stations.

        Uses the latest hour for which all stations have
        complete pollutant observations.
        """

        live = self.engineer_features(live)

        # Find the latest timestamp containing all 37 stations
        # with complete values for all six pollutants.
        complete_hours = (
            live.dropna(subset=self.targets)
            .groupby("time")["location_id"]
            .nunique()
        )

        complete_hours = complete_hours[
            complete_hours == NUM_STATIONS
        ]

        if complete_hours.empty:
            raise ValueError(
                "No timestamp contains complete pollutant "
                "values for all stations."
            )

        latest = complete_hours.index.max()

        X = self.build_model_tensor(
            live,
            latest,
        )

        raw = self.predict_raw(X)

        previous = (
            live[
                live["time"] == latest
            ][
                ["location_id"] + self.targets
            ]
        )

        prediction = self.reconstruct_prediction(
            raw,
            previous,
        )

        forecast_time = (
            latest
            + pd.Timedelta(hours=1)
        )

        result = pd.DataFrame(
            prediction,
            columns=self.targets,
        )

        result.insert(
            0,
            "location_id",
            self.stations,
        )

        result.insert(
            1,
            "time",
            forecast_time,
        )

        return forecast_time, result

    # ========================================================
    # Recursive 72-hour forecast
    # ========================================================

    def forecast_72_hours(
        self,
        live: pd.DataFrame,
        weather: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate a recursive 72-hour forecast.

        Each predicted hour is appended back into the
        historical sequence before the next hour is predicted,
        while forecast weather remains supplied by Open-Meteo.
        """

        live = self.engineer_features(
            live
        )

        complete_hours = (
            live.dropna(subset=self.targets)
            .groupby("time")["location_id"]
            .nunique()
        )

        complete_hours = complete_hours[
            complete_hours == NUM_STATIONS
        ]

        if complete_hours.empty:
            raise ValueError(
                "No complete pollutant hour is available for 72-hour forecasting."
            )

        latest = complete_hours.index.max()

        future_times = pd.date_range(
            latest + pd.Timedelta(hours=1),
            latest + pd.Timedelta(hours=FORECAST_HOURS),
            freq="h",
        )

        future_weather = weather[
            weather["time"].isin(
                future_times
            )
        ].copy()

        # Add station coordinates.
        future_weather = future_weather.merge(
            self.station_metadata[
                [
                    "location_id",
                    "latitude",
                    "longitude",
                ]
            ],
            on="location_id",
            how="left",
        )

        future_weather = future_weather[
            WEATHER_COLUMNS
        ]

        available_hours = (
            future_weather["time"]
            .nunique()
        )

        if available_hours != FORECAST_HOURS:
            raise ValueError(
                f"Expected {FORECAST_HOURS} future "
                f"weather hours, found {available_hours}."
            )

        if (
            future_weather[
                "location_id"
            ].nunique()
            != NUM_STATIONS
        ):
            raise ValueError(
                "Future weather does not cover all 37 stations."
            )

        history = live[live["time"] <= latest].copy()
        forecast_rows = []

        # ====================================================
        # Recursive loop
        # ====================================================

        for forecast_time in future_times:

            # ----------------------------------------------
            # Weather for the current prediction hour
            # ----------------------------------------------

            future_weather_row = future_weather[
                future_weather["time"]
                == forecast_time
            ].copy()

            future_weather_row[
                self.targets
            ] = np.nan

            if len(
                future_weather_row
            ) != NUM_STATIONS:
                raise ValueError(
                    f"Missing weather for "
                    f"{forecast_time}."
                )

            # ----------------------------------------------
            # Add this hour to working history
            # ----------------------------------------------

            working = pd.concat(
                [
                    history,
                    future_weather_row,
                ],
                ignore_index=True,
            )

            working = working.sort_values(
                [
                    "location_id",
                    "time",
                ]
            ).reset_index(drop=True)

            # ----------------------------------------------
            # Feature engineering
            # ----------------------------------------------

            working = self.engineer_features(
                working
            )

            # ----------------------------------------------
            # Build the previous 24-hour input window
            # ----------------------------------------------

            input_end = (
                forecast_time
                - pd.Timedelta(hours=1)
            )

            X = self.build_model_tensor(
                working,
                input_end,
            )

            raw = self.predict_raw(X)

            # ----------------------------------------------
            # Previous pollutant values
            # ----------------------------------------------

            previous_time = (
                history["time"].max()
            )

            previous = (
                history[
                    history["time"]
                    == previous_time
                ][
                    ["location_id"]
                    + self.targets
                ]
            )

            prediction = (
                self.reconstruct_prediction(
                    raw,
                    previous,
                )
            )

            # ----------------------------------------------
            # Create predicted row
            # ----------------------------------------------

            predicted_rows = future_weather_row[
                WEATHER_COLUMNS
            ].copy()

            predicted_rows[
                self.targets
            ] = prediction

            # ----------------------------------------------
            # Feed predictions into next recursive step
            # ----------------------------------------------

            history = pd.concat(
                [
                    history,
                    predicted_rows,
                ],
                ignore_index=True,
            )

            # Keep the pollutant forecast output compact.
            output_rows = predicted_rows[
                ["location_id", "time"]
                + self.targets
            ].copy()

            forecast_rows.append(
                output_rows
            )

        forecast = pd.concat(
            forecast_rows,
            ignore_index=True,
        )

        # ====================================================
        # Final validation
        # ====================================================

        expected_rows = (
            NUM_STATIONS
            * FORECAST_HOURS
        )

        if len(forecast) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} forecast rows, "
                f"got {len(forecast)}."
            )

        if (
            forecast["location_id"]
            .nunique()
            != NUM_STATIONS
        ):
            raise ValueError(
                "72-hour forecast does not contain all stations."
            )

        if (
            forecast["time"]
            .nunique()
            != FORECAST_HOURS
        ):
            raise ValueError(
                "72-hour forecast does not contain all hours."
            )

        if forecast[
            self.targets
        ].isna().any().any():
            raise ValueError(
                "72-hour forecast contains missing pollutant values."
            )

        if not np.isfinite(
            forecast[
                self.targets
            ].to_numpy()
        ).all():
            raise ValueError(
                "72-hour forecast contains non-finite values."
            )

        return forecast

    # ========================================================
    # Complete live pipeline
    # ========================================================

    def run(
        self,
    ) -> Dict[str, object]:
        """
        Run the complete live forecasting pipeline.

        Returns a dictionary containing:
            live data
            next-hour forecast
            72-hour forecast
            metadata
        """

        print("Fetching OpenWeather 48-hour history...")
        pollutant_history = (
            self.fetch_openweather_history(
                hours=48
            )
        )

        print("Fetching OpenWeather current observations...")
        current_pollution = (
            self.fetch_openweather_current()
        )

        print("Fetching Open-Meteo weather...")
        weather = self.fetch_weather(
            past_hours=48,
            forecast_hours=73,
        )

        print("Building live dataset...")
        live = self.build_live_dataset(
            pollutant_history,
            current_pollution,
            weather,
        )

        print("Generating next-hour forecast...")
        next_time, next_hour = (
            self.forecast_next_hour(
                live
            )
        )

        print("Generating recursive 72-hour forecast...")
        forecast_72h = self.forecast_72_hours(
            live,
            weather,
        )

        return {
            "live": live,
            "next_hour_time": next_time,
            "next_hour": next_hour,
            "forecast_72h": forecast_72h,
            "weather": weather,
        }


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    print("PolluSenseAI Live Forecasting Engine")
    print("=" * 60)

    forecaster = LiveForecaster()

    print(
        f"Stations : {len(forecaster.stations)}"
    )

    print(
        f"Features : {len(forecaster.features)}"
    )

    print(
        f"Targets  : {len(forecaster.targets)}"
    )

    print(
        f"Seq len  : {forecaster.seq_len}"
    )

    print(
        f"Edges    : {forecaster.edge_index.shape[1]}"
    )

    print(
        f"Parameters: "
        f"{forecaster.bundle.parameter_count():,}"
    )

    print("\nRunning live pipeline...\n")

    results = forecaster.run()

    live = results["live"]
    next_hour = results["next_hour"]
    forecast_72h = results["forecast_72h"]

    print("\n" + "=" * 60)
    print("LIVE DATA")
    print("=" * 60)

    print(
        f"Rows     : {len(live)}"
    )

    print(
        f"Stations : {live['location_id'].nunique()}"
    )

    print(
        f"Hours    : {live['time'].nunique()}"
    )

    print(
        f"Range    : {live['time'].min()} "
        f"→ {live['time'].max()}"
    )

    print(
        f"Missing  : "
        f"{live.isna().sum().sum()}"
    )

    print("\n" + "=" * 60)
    print("NEXT-HOUR FORECAST")
    print("=" * 60)

    print(
        "Forecast UTC:",
        results["next_hour_time"],
    )

    print(
        next_hour.round(3).to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("72-HOUR FORECAST")
    print("=" * 60)

    print(
        f"Rows     : {len(forecast_72h)}"
    )

    print(
        f"Stations : "
        f"{forecast_72h['location_id'].nunique()}"
    )

    print(
        f"Hours    : "
        f"{forecast_72h['time'].nunique()}"
    )

    print(
        f"Range    : "
        f"{forecast_72h['time'].min()} "
        f"→ "
        f"{forecast_72h['time'].max()}"
    )

    print(
        f"Missing  : "
        f"{forecast_72h.isna().sum().sum()}"
    )

    print("\nFirst forecast hour:")
    print(
        forecast_72h[
            forecast_72h["time"]
            == forecast_72h["time"].min()
        ]
        .head(37)
        .round(3)
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("✓ Live forecasting pipeline validation passed.")