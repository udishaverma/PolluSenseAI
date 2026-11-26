import folium
import streamlit as st
from streamlit_folium import st_folium


AQI_COLORS = {
    "Good": "green",
    "Satisfactory": "yellow",
    "Moderate": "orange",
    "Poor": "red",
    "Very Poor": "darkred",
    "Severe": "black",
}


def _get_color(category):
    return AQI_COLORS.get(category, "gray")


def render_map(stations, selected_station=None):
    m = folium.Map(
        location=[28.5, 77.25],
        zoom_start=9,
        min_zoom=9,
        tiles="OpenStreetMap",
    )

    bounds = [[28.0, 76.5], [29.0, 78.0]]
    m.fit_bounds(bounds, padding=(8, 8))

    for station in stations:
        station_id = station["station_id"]
        name = station["name"]
        lat = float(station["latitude"])
        lon = float(station["longitude"])
        aqi = float(station["aqi"])
        category = station["category"]

        color = _get_color(category)
        is_selected = station_id == selected_station

        tooltip = folium.Tooltip(
            f"""
            <div style="font-family:Arial; font-size:13px;">
                <b>{name}</b><br>
                AQI: <b>{aqi:.0f}</b><br>
                Category: {category}
            </div>
            """,
            sticky=True,
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=11 if is_selected else 7,
            color="black" if is_selected else color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0 if is_selected else 0.9,
            weight=4 if is_selected else 2,
            tooltip=tooltip,
        ).add_to(m)

    result = st_folium(
        m,
        use_container_width=True,
        height=560,
        returned_objects=["last_object_clicked"],
    )

    return result.get("last_object_clicked")