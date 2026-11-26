
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap


UNITS = {
    "co": "mg/m³",
    "no2": "µg/m³",
    "o3": "µg/m³",
    "pm10": "µg/m³",
    "pm25": "µg/m³",
    "so2": "µg/m³",
}


def _attention(
    model, xseq, edge_index, edge_attr, stations, si
):
    model.eval()

    with torch.no_grad():
        B, T, N, Fdim = xseq.shape

        z = xseq.permute(
            0, 2, 1, 3
        ).reshape(B * N, T, Fdim)

        h, _ = model.lstm(z)
        h = h[:, -1].view(B, N, -1)

        e = model.emb(
            torch.arange(N, device=h.device)
        ).unsqueeze(0).expand(B, -1, -1)

        h = torch.cat([h, e], -1).reshape(B * N, -1)

        z = F.elu(
            model.gat1(
                h, edge_index, edge_attr=edge_attr
            )
        )

        z = model.drop(model.n1(z))

        _, (ei, alpha) = model.gat2(
            z,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )

    ei = ei.cpu().numpy()
    alpha = alpha.mean(-1).cpu().numpy()

    rows = [
        (stations[src], float(alpha[j]))
        for j, src in enumerate(ei[0])
        if ei[1, j] == si and src != si
    ]

    if not rows:
        return pd.Series(dtype=float, name="attention")

    return (
        pd.DataFrame(
            rows, columns=["station", "attention"]
        )
        .groupby("station").attention.mean()
        .sort_values(ascending=False)
    )


class _Wrapper(nn.Module):

    def __init__(
        self,
        model,
        base,
        si,
        pi,
        edge_index,
        edge_attr,
        res_mean,
        res_std,
        last_value,
    ):
        super().__init__()

        self.model = model
        self.base = base

        self.si = si
        self.pi = pi

        self.edge_index = edge_index
        self.edge_attr = edge_attr

        self.res_mean = float(res_mean)
        self.res_std = float(res_std)
        self.last_value = float(last_value)

    def forward(self, x):

        z = self.base.unsqueeze(0).repeat(
            x.shape[0], 1, 1, 1
        )

        z[:, -1, self.si, :] = x

        residual_scaled = self.model(
            z,
            self.edge_index,
            self.edge_attr,
        )[:, self.si, self.pi]

        # Convert the explained model output into the final
        # pollutant concentration units.
        prediction = (
            residual_scaled * self.res_std
            + self.res_mean
            + self.last_value
        )

        return prediction.unsqueeze(1)


def explain_prediction(
    model,
    xseq,
    edge_index,
    edge_attr,
    stations,
    features,
    targets,
    station,
    pollutant,
    res_mean,
    res_std,
    last_value,
    background_hours=8,
    top_k=10,
):
    si = stations.index(station)
    pi = targets.index(pollutant)

    model.eval()

    with torch.no_grad():
        raw = model(
            xseq, edge_index, edge_attr
        )[0, si, pi].item()

    # Final predicted concentration in original units.
    prediction = (
        raw * float(res_std[pi])
        + float(res_mean[pi])
        + float(last_value)
    )

    base = xseq[0].detach()
    current = base[-1, si, :].unsqueeze(0)

    wrapper = _Wrapper(
        model,
        base,
        si,
        pi,
        edge_index,
        edge_attr,
        res_mean[pi],
        res_std[pi],
        last_value,
    ).eval()

    background = base[
        -background_hours:, si, :
    ]

    explainer = shap.GradientExplainer(
        wrapper,
        background,
    )

    sv = np.squeeze(
        np.asarray(explainer.shap_values(current))
    )

    names = (
        list(features)
        + [f"{p}_history" for p in targets]
        + [f"{p}_available" for p in targets]
    )

    shap_df = pd.DataFrame({
        "feature": names,
        "shap": sv,
    })

    shap_df["abs_shap"] = shap_df.shap.abs()

    shap_df = shap_df.sort_values(
        "abs_shap",
        ascending=False,
    )

    spatial = _attention(
        model,
        xseq,
        edge_index,
        edge_attr,
        stations,
        si,
    ).head(top_k)

    pos = (
        shap_df[shap_df.shap > 0]
        .sort_values("abs_shap", ascending=False)
        .head(3)
    )

    neg = (
        shap_df[shap_df.shap < 0]
        .sort_values("shap", ascending=True)
        .head(3)
    )

    unit = UNITS[pollutant]

    text = (
        f"For {station}, the GAT-LSTM predicted "
        f"{prediction:.2f} {unit} for {pollutant.upper()}. "
        f"Positive SHAP values indicate features that increased "
        f"the prediction relative to the explainer's background; "
        f"negative values indicate features that decreased it. "
        f"The leading positive contributors were "
        f"{', '.join(pos.feature)}. The leading negative "
        f"contributors were {', '.join(neg.feature)}. "
        f"The strongest incoming spatial-attention weights "
        f"were assigned to {', '.join(spatial.index[:3])}. "
        f"SHAP contributions and spatial attention describe "
        f"model behavior, not physical causation."
    )

    return {
        "station": station,
        "pollutant": pollutant,
        "prediction": float(prediction),
        "unit": unit,
        "shap_value_unit": unit,
        "spatial_attention": [
            {
                "station": str(s),
                "attention": float(v),
            }
            for s, v in spatial.items()
        ],
        "shap": [
            {
                "feature": str(r.feature),
                "value": float(r.shap),
            }
            for _, r in shap_df.head(top_k).iterrows()
        ],
        "top_positive_features": pos.feature.tolist(),
        "top_negative_features": neg.feature.tolist(),
        "natural_language": text,
    }
