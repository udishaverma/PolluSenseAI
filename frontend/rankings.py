import pandas as pd
import streamlit as st

from styles import (
    apply_styles,
    show_loading_screen,
)

from dashboard import (
    get_forecaster,
    get_live_results,
)

from backend.response import build_response


CATEGORY_ICONS = {
    "Good": "🟢",
    "Satisfactory": "🟡",
    "Moderate": "🟠",
    "Poor": "🔴",
    "Very Poor": "🟣",
    "Severe": "🟥",
}


def load_data():
    forecaster = get_forecaster()
    results = get_live_results()

    return build_response(
        results,
        forecaster,
    )


def rankings_dataframe(data):
    rows = []

    for item in data["rankings"]:
        rows.append(
            {
                "Rank": item["rank"],
                "Station": item["station_name"],
                "AQI": int(item["aqi"]),
                "Category": item["category"],
                "Dominant pollutant": (
                    item["dominant_pollutant"].upper()
                ),
                "Station ID": item["station_id"],
            }
        )

    return pd.DataFrame(rows)


def category_icon(category):
    return CATEGORY_ICONS.get(
        category,
        "⚪",
    )


def run_rankings():
    apply_styles()

    # ------------------------------------------------
    # Loading
    # ------------------------------------------------

    loading = show_loading_screen()

    data = load_data()

    loading.empty()

    # ------------------------------------------------
    # Data
    # ------------------------------------------------

    df = rankings_dataframe(data)

    # ------------------------------------------------
    # Page header
    # ------------------------------------------------

    st.markdown(
        '<div class="app-title">Station Rankings</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Compare current air quality across all 37 monitoring stations.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ------------------------------------------------
    # Sorting
    # ------------------------------------------------

    sort_order = st.radio(
        "Sort by",
        [
            "Best → Worst",
            "Worst → Best",
            "Alphabetical",
        ],
        horizontal=True,
        key="ranking_sort",
    )

    if sort_order == "Best → Worst":
        display_df = df.sort_values(
            "AQI",
            ascending=True,
        ).reset_index(drop=True)

    elif sort_order == "Worst → Best":
        display_df = df.sort_values(
            "AQI",
            ascending=False,
        ).reset_index(drop=True)

    else:
        display_df = df.sort_values(
            "Station",
            ascending=True,
        ).reset_index(drop=True)

    # ------------------------------------------------
    # Summary
    # ------------------------------------------------

    best = int(df["AQI"].min())
    worst = int(df["AQI"].max())
    average = df["AQI"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Best AQI",
            best,
        )

    with col2:
        st.metric(
            "Worst AQI",
            worst,
        )

    with col3:
        st.metric(
            "Network average",
            f"{average:.0f}",
        )

    st.write("")

    # ------------------------------------------------
    # Ranking table
    # ------------------------------------------------

    table_df = display_df[
        [
            "Rank",
            "Station",
            "AQI",
            "Category",
            "Dominant pollutant",
        ]
    ].copy()

    table_df["Category"] = table_df["Category"].apply(
        lambda x: f"{category_icon(x)} {x}"
    )

    st.dataframe(
        table_df,
        width="stretch",
        hide_index=True,
        height=650,
    )

    st.write("")

    # ------------------------------------------------
    # Station selector
    # ------------------------------------------------

    st.markdown(
        "### View a station"
    )

    station_lookup = {
        item["station_id"]: item["station_name"]
        for item in data["rankings"]
    }

    station_ids = list(
        station_lookup.keys()
    )

    if (
        "ranking_station_selector"
        not in st.session_state
    ):
        st.session_state.ranking_station_selector = (
            station_ids[0]
        )

    if (
        st.session_state.ranking_station_selector
        not in station_ids
    ):
        st.session_state.ranking_station_selector = (
            station_ids[0]
        )

    selected = st.selectbox(
        "Select station",
        options=station_ids,
        format_func=lambda x: station_lookup[x],
        key="ranking_station_selector",
    )

    selected_info = data["stations"][selected]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current AQI",
            int(selected_info["aqi"]),
        )

    with col2:
        st.metric(
            "Category",
            selected_info["category"],
        )

    with col3:
        st.metric(
            "Dominant pollutant",
            selected_info[
                "dominant_pollutant"
            ].upper(),
        )

    # ------------------------------------------------
    # Open on Dashboard
    # ------------------------------------------------

    if st.button(
        "Open on Dashboard",
        type="primary",
        width="stretch",
        key="open_dashboard",
    ):
        st.session_state.selected_station = selected
        st.session_state.station_selector = selected
        st.session_state.last_map_station = None
        st.session_state.page = "Dashboard"

        st.rerun()

    # ------------------------------------------------
    # Footer
    # ------------------------------------------------

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
    st.set_page_config(
        page_title="PolluSenseAI Rankings",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    run_rankings()