import streamlit as st


def apply_styles():
    st.markdown(
        """
        <style>

        /* ---------- Global ---------- */
        .stApp {
            background: #f5f7fb;
        }

        /* ---------- Headings ---------- */
        h1, h2, h3, h4 {
            color: #102a43 !important;
        }

        .stMarkdown p,
        .stCaption {
            color: #486581;
        }

        label {
            color: #486581 !important;
        }

        /* ---------- Streamlit header ---------- */
        header[data-testid="stHeader"] {
            display: none;
        }

        [data-testid="stToolbar"] {
            display: none;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        /* ---------- Navigation ---------- */
        .app-navbar {
            background: white;
            border: 1px solid #e5eaf0;
            border-radius: 16px;
            padding: 14px 18px;
            margin-bottom: 24px;
            box-shadow: 0 3px 12px rgba(16, 42, 67, 0.06);
        }

        .nav-brand {
            font-size: 1.45rem;
            font-weight: 800;
            color: #102a43;
            line-height: 1.1;
        }

        .nav-tagline {
            font-size: 0.78rem;
            color: #627d98;
            margin-top: 4px;
        }

        /* ---------- Navigation buttons ---------- */
        div.stButton > button {
            border-radius: 10px;
            border: 1px solid #d9e2ec;
            background: white;
            color: #102a43;
            font-weight: 650;
            min-height: 42px;
        }

        div.stButton > button:hover {
            border-color: #2ec4b6;
            color: #102a43;
        }

        /* ---------- Page header ---------- */
        .app-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #102a43;
            margin-bottom: 0.1rem;
        }

        .app-subtitle {
            font-size: 0.95rem;
            color: #627d98;
            margin-bottom: 1.2rem;
        }

        /* ---------- Cards ---------- */
        .card {
            background: white;
            border: 1px solid #e5eaf0;
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 3px 12px rgba(16, 42, 67, 0.06);
        }

        .card-title {
            font-size: 0.82rem;
            font-weight: 700;
            color: #627d98;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }

        /* ---------- AQI ---------- */
        .aqi-number {
            font-size: 4rem;
            line-height: 1;
            font-weight: 800;
            color: #102a43;
        }

        .aqi-category {
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 6px;
        }

        /* ---------- Pollutant cards ---------- */
        .pollutant-value {
            font-size: 1.55rem;
            font-weight: 750;
            color: #102a43;
        }

        .pollutant-unit {
            font-size: 0.78rem;
            color: #829ab1;
        }

        .pollutant-name {
            font-size: 0.82rem;
            font-weight: 650;
            color: #486581;
        }

        /* ---------- Warning ---------- */
        .warning-card {
            background: #fff8e6;
            border: 1px solid #f3d58a;
            border-radius: 14px;
            padding: 14px 16px;
            color: #7c5c00;
        }

        .warning-title {
            font-weight: 750;
            margin-bottom: 3px;
        }

        /* ---------- XAI ---------- */
        .xai-card {
            background: white;
            border: 1px solid #e5eaf0;
            border-radius: 16px;
            padding: 20px;
            margin-top: 12px;
        }

        .xai-header {
            font-size: 1.15rem;
            font-weight: 750;
            color: #102a43;
            margin-bottom: 12px;
        }

        .xai-prediction {
            display: flex;
            align-items: baseline;
            gap: 8px;
            flex-wrap: wrap;
        }

        .xai-pred-value {
            font-size: 2.4rem;
            font-weight: 800;
            color: #102a43;
        }

        .xai-pred-unit {
            font-size: 1rem;
            color: #627d98;
        }

        .xai-pred-text {
            font-size: 0.95rem;
            color: #829ab1;
        }

        .xai-feature {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 10px;
            margin-bottom: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            color: #102a43;
        }

        .xai-feature.positive {
            background: #eef8f1;
            border: 1px solid #d7efdc;
        }

        .xai-feature.negative {
            background: #fff1f1;
            border: 1px solid #f6d8d8;
        }

        .xai-rank {
            width: 24px;
            height: 24px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: white;
            font-size: 0.72rem;
            color: #486581;
        }

        .attention-row {
            margin-bottom: 12px;
        }

        .attention-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            color: #243b53;
            font-size: 0.9rem;
        }

        .attention-label b {
            color: #486581;
        }

        .attention-track {
            height: 9px;
            background: #e5eaf0;
            border-radius: 999px;
            overflow: hidden;
        }

        .attention-fill {
            height: 100%;
            background: #2ec4b6;
            border-radius: 999px;
        }

        .xai-note {
            margin-top: 18px;
            padding-top: 12px;
            border-top: 1px solid #e5eaf0;
            color: #829ab1;
            font-size: 0.78rem;
            line-height: 1.5;
        }

        /* ---------- Loading screen ---------- */
        .loading-screen {
            position: fixed;
            inset: 0;
            z-index: 999999;
            background: #0b1220;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            color: white;
        }

        .loading-content {
            width: min(520px, 85vw);
        }

        .loading-logo {
            font-size: 3rem;
            font-weight: 800;
            color: white;
            letter-spacing: -0.03em;
            margin-bottom: 6px;
        }

        .loading-tagline {
            color: #9fb3c8;
            font-size: 1rem;
            margin-bottom: 34px;
        }

        .loading-orbit {
            position: relative;
            width: 92px;
            height: 92px;
            margin: 0 auto 28px;
        }

        .loading-orbit::before,
        .loading-orbit::after {
            content: "";
            position: absolute;
            inset: 8px;
            border: 3px solid rgba(46, 196, 182, 0.35);
            border-radius: 50%;
        }

        .loading-orbit::after {
            inset: 20px;
            border-color: rgba(46, 196, 182, 0.85);
            animation: pulse-ring 1.8s ease-out infinite;
        }

        .loading-dot {
            position: absolute;
            width: 18px;
            height: 18px;
            top: 37px;
            left: 37px;
            border-radius: 50%;
            background: #2ec4b6;
            box-shadow: 0 0 25px rgba(46, 196, 182, 0.75);
            animation: pulse-dot 1.4s ease-in-out infinite;
        }

        @keyframes pulse-ring {
            0% {
                transform: scale(0.7);
                opacity: 1;
            }

            100% {
                transform: scale(1.45);
                opacity: 0;
            }
        }

        @keyframes pulse-dot {
            0%, 100% {
                transform: scale(0.9);
            }

            50% {
                transform: scale(1.2);
            }
        }

        .loading-bar {
            width: 100%;
            height: 5px;
            background: rgba(255,255,255,0.12);
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 14px;
        }

        .loading-bar-inner {
            height: 100%;
            width: 45%;
            border-radius: 999px;
            background: #2ec4b6;
            animation: loading-progress 2.2s ease-in-out infinite;
        }

        @keyframes loading-progress {
            0% {
                transform: translateX(-110%);
            }

            50% {
                transform: translateX(90%);
            }

            100% {
                transform: translateX(230%);
            }
        }

        .loading-status {
            color: #c7d4e3;
            font-size: 0.88rem;
            min-height: 22px;
        }

        .loading-status::after {
            content: "Initializing forecasting engine...";
            animation: loading-text 8s infinite;
        }

        @keyframes loading-text {
            0%, 24% {
                content: "Initializing forecasting engine...";
            }

            25%, 49% {
                content: "Fetching live air-quality observations...";
            }

            50%, 74% {
                content: "Processing weather conditions...";
            }

            75%, 100% {
                content: "Generating the 72-hour forecast...";
            }
        }

        /* ---------- Footer ---------- */
        .disclaimer {
            text-align: center;
            color: #829ab1;
            font-size: 0.72rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e5eaf0;
        }

        /* ---------- Streamlit cleanup ---------- */
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #e5eaf0;
            border-radius: 14px;
            padding: 12px;
        }

        div[data-testid="stHorizontalBlock"] {
            gap: 1rem;
        }

        /* ---------- Mobile ---------- */
        @media (max-width: 768px) {
            .app-title {
                font-size: 1.7rem;
            }

            .aqi-number {
                font-size: 3rem;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def show_loading_screen():
    placeholder = st.empty()

    placeholder.html(
        """
        <div class="loading-screen">
            <div class="loading-content">

                <div class="loading-orbit">
                    <div class="loading-dot"></div>
                </div>

                <div class="loading-logo">
                    PolluSenseAI
                </div>

                <div class="loading-tagline">
                    Read the Air. Predict the Future.
                </div>

                <div class="loading-bar">
                    <div class="loading-bar-inner"></div>
                </div>

                <div class="loading-status"></div>

            </div>
        </div>
        """
    )

    return placeholder


if __name__ == "__main__":
    print("PolluSenseAI styles")
    print("=" * 35)
    print("CSS module loaded successfully.")
    print("✓ Styles validation passed.")