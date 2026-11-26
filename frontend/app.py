import streamlit as st

from styles import apply_styles
from dashboard import run_dashboard
from rankings import run_rankings


def render_navbar():
    st.html(
        """
        <div class="app-navbar">
            <div class="nav-brand">
                PolluSenseAI
            </div>

            <div class="nav-tagline">
                Read the Air. Predict the Future.
            </div>
        </div>
        """
    )

    col1, col2, _ = st.columns(
        [1, 1, 4],
        vertical_alignment="center",
    )

    with col1:
        if st.button(
            "Dashboard",
            width="stretch",
            key="nav_dashboard",
        ):
            st.session_state.page = "Dashboard"
            st.rerun()

    with col2:
        if st.button(
            "Station Rankings",
            width="stretch",
            key="nav_rankings",
        ):
            st.session_state.page = "Rankings"
            st.rerun()


def main():
    st.set_page_config(
        page_title="PolluSenseAI",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    apply_styles()

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    render_navbar()

    if st.session_state.page == "Rankings":
        run_rankings()
    else:
        run_dashboard()


if __name__ == "__main__":
    main()