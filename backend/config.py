from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# Project paths
# ============================================================

# backend/config.py
# parent -> backend/
# parent.parent -> PolluSenseAI/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
GRAPH_DIR = PROJECT_ROOT / "graph"
MODELS_DIR = PROJECT_ROOT / "models"
XAI_DIR = PROJECT_ROOT / "xai"
BACKEND_DIR = PROJECT_ROOT / "backend"


# ============================================================
# Environment variables
# ============================================================

# Load .env from the project root
load_dotenv(PROJECT_ROOT / ".env")

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


# ============================================================
# Model configuration
# ============================================================

MODEL_PATH = MODELS_DIR / "best_improved_gat_lstm.pt"
SCALER_PATH = MODELS_DIR / "feature_scaler.pkl"
MODEL_CONFIG_PATH = MODELS_DIR / "gat_lstm_config.pkl"

EDGE_INDEX_PATH = GRAPH_DIR / "graph_edge_index.npy"
STATION_METADATA_PATH = GRAPH_DIR / "station_metadata_final.csv"


# ============================================================
# Forecast configuration
# ============================================================

SEQUENCE_LENGTH = 24
FORECAST_HOURS = 72

NUM_STATIONS = 37
NUM_FEATURES = 34
NUM_INPUT_CHANNELS = 46
NUM_TARGETS = 6


# ============================================================
# Pollutants
# ============================================================

POLLUTANTS = [
    "co",
    "no2",
    "o3",
    "pm10",
    "pm25",
    "so2",
]


# ============================================================
# Station IDs
# ============================================================

STATION_IDS = [f"ST{i:02d}" for i in range(1, 38)]


# ============================================================
# API configuration
# ============================================================

OPENWEATHER_BASE_URL = (
    "https://api.openweathermap.org/data/2.5/air_pollution"
)

OPENWEATHER_HISTORY_URL = (
    "https://api.openweathermap.org/data/2.5/air_pollution/history"
)

OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# Runtime settings
# ============================================================

DEVICE = "cpu"


# ============================================================
# Validation
# ============================================================

def validate_config() -> None:
    """
    Validate that required project files and configuration exist.
    """

    required_paths = {
        "model": MODEL_PATH,
        "scaler": SCALER_PATH,
        "model_config": MODEL_CONFIG_PATH,
        "edge_index": EDGE_INDEX_PATH,
        "station_metadata": STATION_METADATA_PATH,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]

    if missing:
        message = "Missing required project files:\n" + "\n".join(missing)
        raise FileNotFoundError(message)

    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY was not found. "
            "Add it to the project-root .env file."
        )


if __name__ == "__main__":
    print("PolluSenseAI backend configuration")
    print("=" * 45)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Scaler       : {SCALER_PATH}")
    print(f"Graph        : {EDGE_INDEX_PATH}")
    print(f"Stations     : {NUM_STATIONS}")
    print(f"Features     : {NUM_FEATURES}")
    print(f"Input shape  : ({SEQUENCE_LENGTH}, {NUM_STATIONS}, {NUM_INPUT_CHANNELS})")
    print(f"Forecast     : {FORECAST_HOURS} hours")
    print(f"API key      : {'Loaded' if OPENWEATHER_API_KEY else 'Missing'}")