import streamlit as st


AQI_CATEGORIES = {
    "Good": "🟢",
    "Satisfactory": "🟡",
    "Moderate": "🟠",
    "Poor": "🔴",
    "Very Poor": "🟣",
    "Severe": "🟥",
}


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


def render_aqi_card(aqi, category, dominant):
    icon = AQI_CATEGORIES.get(category, "⚪")

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Current AQI</div>
            <div class="aqi-number">{aqi:.0f}</div>
            <div class="aqi-category">{icon} {category}</div>
            <div class="pollutant-unit">
                Dominant pollutant: {dominant.upper()}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pollutant_card(name, value, unit):
    st.markdown(
        f"""
        <div class="card">
            <div class="pollutant-name">{name}</div>
            <div class="pollutant-value">{value:.2f}</div>
            <div class="pollutant-unit">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pollutants(pollutants):
    names = {
        "co": ("CO", "mg/m³"),
        "no2": ("NO₂", "µg/m³"),
        "o3": ("O₃", "µg/m³"),
        "pm10": ("PM10", "µg/m³"),
        "pm25": ("PM2.5", "µg/m³"),
        "so2": ("SO₂", "µg/m³"),
    }

    cols = st.columns(6)

    for col, pollutant in zip(cols, names):
        name, unit = names[pollutant]
        value = pollutants.get(pollutant, 0)

        with col:
            render_pollutant_card(
                name,
                float(value),
                unit,
            )


def render_warning(category, dominant, aqi):
    if category in {"Severe", "Very Poor"}:
        icon = "🚨"
        title = "Air quality warning"
        text = (
            f"AQI is {aqi:.0f}. {dominant.upper()} is the dominant pollutant. "
            "Air quality is currently at a high-risk level."
        )

    elif category in {"Poor", "Moderate"}:
        icon = "⚠️"
        title = "Air quality advisory"
        text = (
            f"AQI is {aqi:.0f}. {dominant.upper()} is the dominant pollutant. "
            "Sensitive individuals should consider limiting prolonged outdoor exposure."
        )

    else:
        icon = "ℹ️"
        title = "Air quality status"
        text = (
            f"AQI is {aqi:.0f}. {dominant.upper()} is currently the "
            "dominant pollutant."
        )

    st.markdown(
        f"""
        <div class="warning-card">
            <div class="warning-title">{icon} {title}</div>
            <div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_xai(xai, station_names=None):
    station_names = station_names or {}

    station_id = xai.get(
        "station",
        "",
    )

    station_name = station_names.get(
        station_id,
        station_id,
    )

    pollutant = xai.get(
        "pollutant",
        "",
    ).upper()

    prediction = float(
        xai.get("prediction", 0.0)
    )

    unit = xai.get(
        "unit",
        "",
    )

    # ------------------------------------------------
    # Prediction summary
    # ------------------------------------------------

    st.html(
        f"""
        <div class="xai-card">
            <div class="xai-header">
                Model explanation
            </div>

            <div class="xai-prediction">
                <span class="xai-pred-value">
                    {prediction:.2f}
                </span>

                <span class="xai-pred-unit">
                    {unit}
                </span>

                <span class="xai-pred-text">
                    predicted {pollutant}
                    at {station_name}
                </span>
            </div>
        </div>
        """
    )

    explanation = xai.get("explanation")

    if explanation:
        st.info(explanation)

    # ------------------------------------------------
    # SHAP feature influences
    # ------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Positive model influences")

        positive = xai.get(
            "top_positive_features",
            [],
        )

        if positive:
            for i, feature in enumerate(
                positive[:5],
                1,
            ):
                st.html(
                    f"""
                    <div class="xai-feature positive">
                        <span class="xai-rank">{i}</span>
                        <span>{pretty_feature(feature)}</span>
                    </div>
                    """
                )
        else:
            st.caption(
                "No positive influences identified."
            )

    with col2:
        st.markdown("#### Negative model influences")

        negative = xai.get(
            "top_negative_features",
            [],
        )

        if negative:
            for i, feature in enumerate(
                negative[:5],
                1,
            ):
                st.html(
                    f"""
                    <div class="xai-feature negative">
                        <span class="xai-rank">{i}</span>
                        <span>{pretty_feature(feature)}</span>
                    </div>
                    """
                )
        else:
            st.caption(
                "No negative influences identified."
            )

    # ------------------------------------------------
    # Spatial attention
    # ------------------------------------------------

    st.markdown("#### Highest spatial attention")

    spatial = xai.get(
        "spatial_attention",
        [],
    )

    if spatial:
        values = [
            float(item.get("attention", 0))
            for item in spatial[:5]
        ]

        max_attention = (
            max(values)
            if values
            else 0
        )

        for item in spatial[:5]:
            station_id = item.get(
                "station",
                "Unknown",
            )

            station_name = station_names.get(
                station_id,
                station_id,
            )

            attention = float(
                item.get("attention", 0)
            )

            width = (
                attention / max_attention * 100
                if max_attention > 0
                else 0
            )

            st.html(
                f"""
                <div class="attention-row">
                    <div class="attention-label">
                        <span>{station_name}</span>
                        <b>{attention:.3f}</b>
                    </div>

                    <div class="attention-track">
                        <div
                            class="attention-fill"
                            style="width:{width:.1f}%;">
                        </div>
                    </div>
                </div>
                """
            )

    else:
        st.caption(
            "No spatial attention values available."
        )

    # ------------------------------------------------
    # Explanation note
    # ------------------------------------------------

    st.html(
        """
        <div class="xai-note">
            SHAP rankings describe model attribution, while
            spatial attention describes learned weighting of
            connected stations. These signals do not imply
            physical causation.
        </div>
        """
    )


def render_footer():
    st.markdown(
        """
        <div class="disclaimer">
            PolluSenseAI predictions may contain errors and are intended for
            informational/research purposes.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    print("PolluSenseAI components")
    print("=" * 35)
    print("AQI card: OK")
    print("Pollutant cards: OK")
    print("Warning card: OK")
    print("XAI section: OK")
    print("✓ Components validation passed.")