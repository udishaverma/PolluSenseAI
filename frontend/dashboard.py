import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from styles import (
    apply_styles,
    show_loading_screen,
)

from map_view import render_map

from components import (
    render_aqi_card,
    render_pollutants,
    render_warning,
    render_xai,
    render_footer,
)

from backend.forecasting import LiveForecaster

from backend.response import build_response

from backend.xai import (
    explain_station,
    natural_language_summary,
)


@st.cache_resource
def get_forecaster():
    return LiveForecaster()


@st.cache_data(ttl=1800)
def get_live_results():
    forecaster = get_forecaster()
    return forecaster.run()


def station_list(data):
    return [
        {
            "station_id": s["station_id"],
            "name": s["station_name"],
            "latitude": s["latitude"],
            "longitude": s["longitude"],
            "aqi": s["aqi"],
            "category": s["category"],
        }
        for s in data["stations"].values()
    ]


def find_station_from_click(
    clicked,
    stations,
):
    if not clicked:
        return None

    lat = clicked.get("lat")
    lng = clicked.get("lng")

    if lat is None or lng is None:
        return None

    nearest = min(
        stations,
        key=lambda s: (
            (
                float(s["latitude"])
                - lat
            ) ** 2
            +
            (
                float(s["longitude"])
                - lng
            ) ** 2
        ),
    )

    return nearest["station_id"]


def update_selected_from_dropdown():
    st.session_state.selected_station = (
        st.session_state.station_selector
    )


def forecast_chart(
    data,
    station_id,
):
    forecast = data["forecasts"][
        station_id
    ]

    sampled = forecast[::9]

    times = [
        pd.to_datetime(
            item["time_utc"]
        ).tz_convert(
            "Asia/Kolkata"
        )
        for item in sampled
    ]

    aqi = [
        item["aqi"]
        for item in sampled
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times,
            y=aqi,
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=7),
            hovertemplate=(
                "<b>%{x|%d %b, %I:%M %p}</b>"
                "<br>AQI: %{y}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(
            l=55,
            r=20,
            t=20,
            b=55,
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            color="#102a43",
            size=12,
        ),
        xaxis=dict(
            title=dict(
                text="Forecast time (IST)",
                font=dict(
                    color="#486581",
                    size=12,
                ),
            ),
            tickfont=dict(
                color="#486581",
                size=11,
            ),
            tickformat="%I:%M<br>%d %b",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(
                text="AQI",
                font=dict(
                    color="#486581",
                    size=12,
                ),
            ),
            tickfont=dict(
                color="#486581",
                size=11,
            ),
            gridcolor="#d9e2ec",
            gridwidth=1,
            zeroline=False,
        ),
    )

    return fig


def get_station_xai(
    forecaster,
    live,
    station_id,
    pollutant,
):
    cache = st.session_state.setdefault(
        "xai_cache",
        {},
    )

    key = (
        f"{station_id}_{pollutant}"
    )

    if key not in cache:
        xai = explain_station(
            forecaster,
            live,
            station=station_id,
            pollutant=pollutant,
        )

        cache[key] = xai

    return cache[key]


def run_dashboard():
    apply_styles()

    # ------------------------------------------------
    # Live backend
    # ------------------------------------------------

    forecaster = get_forecaster()

    loading = show_loading_screen()

    results = get_live_results()

    loading.empty()

    live = results["live"]

    data = build_response(
        results,
        forecaster,
    )

    stations = station_list(data)

    station_ids = [
        s["station_id"]
        for s in stations
    ]

    station_names = {
        s["station_id"]: s["name"]
        for s in stations
    }

    # ------------------------------------------------
    # Session state
    # ------------------------------------------------

    if (
        "selected_station"
        not in st.session_state
    ):
        st.session_state.selected_station = "ST01"

    if (
        "station_selector"
        not in st.session_state
    ):
        st.session_state.station_selector = "ST01"

    if (
        "last_map_station"
        not in st.session_state
    ):
        st.session_state.last_map_station = None

    if (
        st.session_state.selected_station
        not in station_ids
    ):
        st.session_state.selected_station = "ST01"

    if (
        st.session_state.station_selector
        not in station_ids
    ):
        st.session_state.station_selector = (
            st.session_state.selected_station
        )

    # ------------------------------------------------
    # Header
    # ------------------------------------------------

    st.markdown(
        '<div class="app-title">'
        'PolluSenseAI'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Read the Air. Predict the Future.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------
    # Map
    # ------------------------------------------------

    st.markdown(
        "### Delhi-NCR Air Quality"
    )

    clicked = render_map(
        stations,
        st.session_state.selected_station,
    )

    clicked_station = (
        find_station_from_click(
            clicked,
            stations,
        )
    )

    if (
        clicked_station
        and clicked_station
        != st.session_state.last_map_station
    ):
        st.session_state.last_map_station = (
            clicked_station
        )

        st.session_state.selected_station = (
            clicked_station
        )

        st.session_state.station_selector = (
            clicked_station
        )

        st.rerun()

    # ------------------------------------------------
    # Station selector
    # ------------------------------------------------

    st.selectbox(
        "Select station",
        options=station_ids,
        format_func=lambda x: station_names[x],
        key="station_selector",
        on_change=update_selected_from_dropdown,
    )

    selected_station = (
        st.session_state.selected_station
    )

    # ------------------------------------------------
    # Selected station
    # ------------------------------------------------

    station = data["stations"][
        selected_station
    ]

    name = station["station_name"]
    aqi = float(station["aqi"])
    category = station["category"]
    dominant = station[
        "dominant_pollutant"
    ]

    pollutants = {
        key: value["value"]
        for key, value
        in station["pollutants"].items()
    }

    current_time = pd.to_datetime(
        station["forecast_time_utc"]
    ).tz_convert(
        "Asia/Kolkata"
    )

    st.markdown(
        f"""
        <h2 style="margin-bottom:0;">
            {name}
        </h2>

        <div style="color:#829ab1;">
            Current model time:
            {current_time.strftime(
                "%d %b %Y, %I:%M %p IST"
            )}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # ------------------------------------------------
    # AQI
    # ------------------------------------------------

    render_aqi_card(
        aqi,
        category,
        dominant,
    )

    st.write("")

    # ------------------------------------------------
    # Pollutants
    # ------------------------------------------------

    render_pollutants(
        pollutants
    )

    st.write("")

    # ------------------------------------------------
    # Warning
    # ------------------------------------------------

    render_warning(
        category,
        dominant,
        aqi,
    )

    st.write("")

    # ------------------------------------------------
    # Forecast
    # ------------------------------------------------

    st.markdown(
        "### 72-Hour AQI Forecast"
    )

    st.plotly_chart(
        forecast_chart(
            data,
            selected_station,
        ),
        width="stretch",
    )

    st.caption(
        "Forecast shown at selected intervals "
        "across the next 72 hours."
    )

    # ------------------------------------------------
    # XAI
    # ------------------------------------------------

    st.markdown(
        "### Why is the model predicting this AQI?"
    )

    xai_pollutant = dominant

    with st.spinner(
        f"Generating explanation for "
        f"{xai_pollutant.upper()}..."
    ):
        xai = get_station_xai(
            forecaster,
            live,
            selected_station,
            xai_pollutant,
        )

    xai["explanation"] = (
        natural_language_summary(
            xai,
            station_names,
        )
    )

    render_xai(
        xai,
        station_names,
    )

    # ------------------------------------------------
    # Footer
    # ------------------------------------------------

    render_footer()


if __name__ == "__main__":
    st.set_page_config(
        page_title="PolluSenseAI",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    run_dashboard()