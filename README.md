# PolluSenseAI

**Spatiotemporal GAT-LSTM-Based Air-Quality Forecasting and Explainable AQI Prediction**

PolluSenseAI is a research-oriented air-quality forecasting dashboard
for 37 monitoring stations across Delhi-NCR. It combines temporal
learning with an LSTM and spatial learning with Graph Attention
Network (GATv2) layers.

## Features

- Forecasts six pollutants: CO, NO2, O3, PM10, PM2.5 and SO2.
- Derives pollutant sub-indices and a model-derived AQI using CPCB
  breakpoint values for the six pollutants supported by the model.
- Retrieves recent pollutant observations from OpenWeather.
- Retrieves hourly meteorological data from Open-Meteo.
- Produces recursive 72-hour pollutant forecasts.
- Provides SHAP-based feature attribution and GAT attention visualizations.
- Displays monitoring stations on an interactive map and provides
  station rankings.

> **AQI scope:** The model predicts six pollutants. The complete
> CPCB National AQI framework also includes NH3 and Pb. Therefore,
> the dashboard's AQI is derived from the six modeled pollutants and
> should not be described as a complete eight-pollutant official
> CPCB AQI when NH3 and Pb are unavailable.

## Model architecture

- Input sequence: 24 hours × 37 stations × 46 channels.
- 34 engineered features, 6 pollutant-history channels and
  6 availability-mask channels.
- Two-layer LSTM with hidden size 128.
- Learned station embedding of dimension 16.
- Two GATv2 layers, with four attention heads in the first layer.
- Edge distance supplied as an edge attribute.
- Residual connection, layer normalization and dropout.
- Six pollutant-specific regression heads.
- Trainable parameters: 344,150.

The dashboard loads the original trained GAT-LSTM checkpoint.
It does not train the model during application startup.

## Repository layout

```text
PolluSenseAI/
├── backend/
│   ├── __init__.py
│   ├── aqi.py
│   ├── config.py
│   ├── data.py
│   ├── forecasting.py
│   ├── model.py
│   ├── response.py
│   └── xai.py
├── frontend/
│   ├── app.py
│   ├── components.py
│   ├── dashboard.py
│   ├── map_view.py
│   ├── rankings.py
│   └── styles.py
├── graph/
│   ├── edge_index_final.csv
│   ├── graph_edge_index.npy
│   └── station_metadata_final.csv
├── models/
│   ├── best_improved_gat_lstm.pt
│   ├── feature_scaler.pkl
│   └── gat_lstm_config.pkl
├── scripts/
│   └── xai/
│       └── pollusense_xai.py
├── notebooks/
├── stations.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile.vercel
├── vercel.json
├── requirements.txt
├── requirements-dev.txt
└── README.md