# ============================================================
# backend/model.py
# PolluSenseAI - Trained Residual GAT-LSTM Loader
# ============================================================

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from torch_geometric.nn import GATv2Conv

try:
    from .config import (
        DEVICE,
        MODEL_PATH,
        MODEL_CONFIG_PATH,
        EDGE_INDEX_PATH,
        STATION_METADATA_PATH,
        NUM_STATIONS,
        NUM_INPUT_CHANNELS,
    )
except ImportError:
    from config import (
        DEVICE,
        MODEL_PATH,
        MODEL_CONFIG_PATH,
        EDGE_INDEX_PATH,
        STATION_METADATA_PATH,
        NUM_STATIONS,
        NUM_INPUT_CHANNELS,
    )


class ResidualGATLSTM(nn.Module):
    """
    Exact model architecture used in Notebook 7.

    Architecture:
        24-hour sequence
            ↓
        2-layer LSTM, hidden=128
            ↓
        station embedding, dim=16
            ↓
        concatenate → 144 dimensions
            ↓
        GATv2, 4 heads → 128 dimensions
            ↓
        LayerNorm + Dropout
            ↓
        GATv2 → 128 dimensions
            ↓
        Residual connection + LayerNorm + Dropout
            ↓
        6 pollutant-specific heads
    """

    def __init__(
        self,
        input_dim: int,
        num_stations: int,
        num_targets: int = 6,
        hidden: int = 128,
        heads: int = 4,
        embedding_dim: int = 16,
        drop: float = 0.2,
    ):
        super().__init__()

        self.num_stations = num_stations
        self.num_targets = num_targets

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=drop,
        )

        # Learned station identity embedding
        self.emb = nn.Embedding(
            num_embeddings=num_stations,
            embedding_dim=embedding_dim,
        )

        # Spatial graph attention
        self.gat1 = GATv2Conv(
            hidden + embedding_dim,
            hidden // heads,
            heads=heads,
            edge_dim=1,
        )

        self.gat2 = GATv2Conv(
            hidden,
            hidden,
            edge_dim=1,
        )

        # Normalization and dropout
        self.n1 = nn.LayerNorm(hidden)
        self.n2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(drop)

        # Six pollutant-specific prediction heads
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, 64),
                    nn.SiLU(),
                    nn.Dropout(drop),
                    nn.Linear(64, 1),
                )
                for _ in range(num_targets)
            ]
        )

    def forward(
        self,
        x_grid: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_grid:
            Tensor of shape [B, T, N, F]

        edge_index:
            Batched graph connectivity of shape [2, E_total]

        edge_attr:
            Batched edge distances of shape [E_total, 1]

        Returns
        -------
        Tensor of shape [B, N, 6]
        """

        B, T, N, Fdim = x_grid.shape

        # --------------------------------------------------------
        # LSTM operates independently on each station
        # --------------------------------------------------------

        # [B, T, N, F]
        #     ↓
        # [B, N, T, F]
        #     ↓
        # [B*N, T, F]
        x = (
            x_grid
            .permute(0, 2, 1, 3)
            .reshape(B * N, T, Fdim)
        )

        h, _ = self.lstm(x)

        # Last timestep representation
        # [B*N, T, 128] → [B*N, 128]
        h = h[:, -1]

        # [B, N, 128]
        h = h.view(B, N, -1)

        # --------------------------------------------------------
        # Station embeddings
        # --------------------------------------------------------

        station_ids = torch.arange(
            self.num_stations,
            device=h.device,
        )

        e = self.emb(station_ids)

        # [N, 16] → [B, N, 16]
        e = e.expand(B, -1, -1)

        # [B, N, 128] + [B, N, 16]
        #              ↓
        #          [B, N, 144]
        h = torch.cat([h, e], dim=-1)

        # Flatten station dimension for PyG
        # [B, N, 144] → [B*N, 144]
        h = h.reshape(B * N, -1)

        # --------------------------------------------------------
        # Batched graph offset
        # --------------------------------------------------------

        offsets = torch.arange(B, device=h.device) * N

        batched_edges = torch.cat(
        [edge_index + offset for offset in offsets],
        dim=1,
        )

        batched_edge_attr = edge_attr.repeat(B, 1)



        # --------------------------------------------------------
        # GAT layer 1
        # --------------------------------------------------------

        z = F.elu(self.gat1(h, batched_edges, edge_attr=batched_edge_attr,))

        z = self.drop(self.n1(z))

        # --------------------------------------------------------
        # GAT layer 2
        # --------------------------------------------------------

        z = self.gat2(z,batched_edges, edge_attr=batched_edge_attr,)

        # Residual connection
        z = self.n2(
            z + h[:, : z.shape[1]]
        )

        z = self.drop(z)

        # [B*N, 128] → [B, N, 128]
        z = z.view(B, N, -1)

        # --------------------------------------------------------
        # Pollutant-specific heads
        # --------------------------------------------------------

        predictions = [
            head(z)
            for head in self.heads
        ]

        # [B, N, 1] × 6
        #        ↓
        # [B, N, 6]
        return torch.cat(predictions, dim=-1)


class ModelBundle:
    """
    Loads the trained model and all artifacts required for
    inference.
    """

    def __init__(self):
        self.device = torch.device(DEVICE)

        # --------------------------------------------------------
        # Load model configuration
        # --------------------------------------------------------

        self.config = joblib.load(MODEL_CONFIG_PATH)

        self.features = self.config["features"]
        self.targets = self.config["targets"]
        self.seq_len = self.config["seq_len"]
        self.stations = self.config["station_list"]

        self.fill_values = self.config["fill_values"]
        self.target_mean = np.asarray(
            self.config["target_mean"],
            dtype=np.float32,
        )
        self.target_std = np.asarray(
            self.config["target_std"],
            dtype=np.float32,
        )

        self.res_mean = np.asarray(
            self.config["res_mean"],
            dtype=np.float32,
        )
        self.res_std = np.asarray(
            self.config["res_std"],
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # Load feature scaler
        # --------------------------------------------------------

        self.feature_scaler = joblib.load(
            MODEL_PATH.parent / "feature_scaler.pkl"
        )

        # --------------------------------------------------------
        # Load station metadata
        # --------------------------------------------------------

        import pandas as pd

        self.station_metadata = pd.read_csv(
            STATION_METADATA_PATH
        )

        # Preserve model station ordering
        self.station_metadata["location_id"] = pd.Categorical(
            self.station_metadata["location_id"],
            categories=self.stations,
            ordered=True,
        )

        self.station_metadata = (
            self.station_metadata
            .sort_values("location_id")
            .reset_index(drop=True)
        )

        # --------------------------------------------------------
        # Load graph
        # --------------------------------------------------------

        self.edge_index = torch.from_numpy(
            np.load(EDGE_INDEX_PATH)
        ).long()

        # Distances are stored in station metadata / edge CSV,
        # so we load the final edge table for edge attributes.
        edge_path = EDGE_INDEX_PATH.parent / "edge_index_final.csv"
        edge_df = pd.read_csv(edge_path)

        station_to_index = {
            station: i
            for i, station in enumerate(self.stations)
        }

        self.edge_index = torch.tensor(
            np.stack(
                [
                    edge_df["source"]
                    .map(station_to_index)
                    .to_numpy(),
                    edge_df["target"]
                    .map(station_to_index)
                    .to_numpy(),
                ]
            ),
            dtype=torch.long,
        )

        self.edge_attr = torch.tensor(
            edge_df["distance_km"].to_numpy(),
            dtype=torch.float32,
        ).view(-1, 1)

        # --------------------------------------------------------
        # Create model
        # --------------------------------------------------------

        self.model = ResidualGATLSTM(
            input_dim=NUM_INPUT_CHANNELS,
            num_stations=NUM_STATIONS,
            num_targets=len(self.targets),
        ).to(self.device)

        # --------------------------------------------------------
        # Load trained weights
        # --------------------------------------------------------

        state_dict = torch.load(
            MODEL_PATH,
            map_location=self.device,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(
        self,
        x_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Run one model prediction.

        Parameters
        ----------
        x_grid:
            NumPy array of shape [B, 24, 37, 46]

        Returns
        -------
        NumPy array of shape [B, 37, 6]
        """

        x = torch.tensor(
            x_grid,
            dtype=torch.float32,
            device=self.device,
        )

        edge_index = self.edge_index.to(self.device)
        edge_attr = self.edge_attr.to(self.device)

        with torch.no_grad():
            prediction = self.model(
                x_grid=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        return prediction.cpu().numpy()

    def parameter_count(self) -> int:
        """Return number of trainable parameters."""
        return sum(
            p.numel()
            for p in self.model.parameters()
            if p.requires_grad
        )


# ============================================================
# Quick standalone test
# ============================================================

if __name__ == "__main__":
    print("Loading PolluSenseAI model...")
    print("=" * 50)

    bundle = ModelBundle()

    print(f"Stations       : {len(bundle.stations)}")
    print(f"Features       : {len(bundle.features)}")
    print(f"Targets        : {len(bundle.targets)}")
    print(f"Sequence length: {bundle.seq_len}")
    print(f"Edges          : {bundle.edge_index.shape[1]}")
    print(f"Parameters     : {bundle.parameter_count():,}")
    print(f"Device         : {bundle.device}")

    assert len(bundle.stations) == 37
    assert len(bundle.features) == 34
    assert len(bundle.targets) == 6
    assert bundle.seq_len == 24
    assert bundle.edge_index.shape[1] == 169
    assert bundle.parameter_count() == 344_150

    print("=" * 50)
    print("✓ Model loaded and validated successfully.")