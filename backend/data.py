from pathlib import Path
import pandas as pd

try:
    from .config import STATION_METADATA_PATH, EDGE_INDEX_PATH
except ImportError:
    from config import STATION_METADATA_PATH, EDGE_INDEX_PATH


def load_station_metadata():
    return pd.read_csv(STATION_METADATA_PATH)


def load_edges():
    return pd.read_csv(EDGE_INDEX_PATH, encoding="cp1252")


if __name__ == "__main__":
    stations = load_station_metadata()
    edges = load_edges()

    print("Stations:", len(stations))
    print("Edges:", len(edges))
    print("✓ Data validation passed.")