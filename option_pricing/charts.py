"""Plotly chart builders for model outputs."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .models import (
    EPS, black_scholes_merton, black_scholes_price_vectorized,
    build_binomial_model, explicit_stability_coefficients, payoff,
    probability_field, risk_neutral_density, stability_map_data,
)

PLOT_TEMPLATE = "plotly_dark"
COLORSCALE_MAIN = "Turbo"
COLORSCALE_PROB = "Viridis"
COLORSCALE_ERROR = "RdBu"


def plot_layout(fig: go.Figure, title: str, height: int = 520) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=45, r=35, t=70, b=45),
        font=dict(family="Inter, Arial, sans-serif", color="#E8EEF8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def make_price_comparison_figure(price_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=price_df["Method"],
            y=price_df["Price"],
            text=[f"{x:.4f}" if np.isfinite(x) else "NA" for x in price_df["Price"]],
            textposition="outside",
            marker=dict(color=price_df["Price"], colorscale=COLORSCALE_MAIN, line=dict(width=1, color="rgba(255,255,255,.4)")),
            hovertemplate="%{x}<br>Price: %{y:.6f}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Option value")
    return plot_layout(fig, "Cross-method price comparison", height=460)


def make_payoff_value_figure(S0: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> go.Figure:
    S_grid = np.linspace(max(0.01, 0.05 * S0), 2.5 * S0, 220)
    intrinsic = payoff(S_grid, K, option_type)
    bsm = black_scholes_price_vectorized(S_grid, K, T, r, q, sigma, option_type)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_grid, y=intrinsic, mode="lines", name="Payoff at maturity", line=dict(dash="dash", width=2)))
    fig.add_trace(go.Scatter(x=S_grid, y=bsm, mode="lines", name="Black-Scholes (European)", line=dict(width=4)))
    fig.add_vline(x=S0, line_dash="dot", annotation_text=f"S0={S0:.2f}")
    fig.add_vline(x=K, line_dash="dash", annotation_text=f"K={K:.2f}")
    fig.update_xaxes(title="Underlying price S")
    fig.update_yaxes(title="Value")
    return plot_layout(fig, "Payoff versus European Black-Scholes value", height=500)


def make_lognormal_distribution_figure(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> go.Figure:
    S_grid = np.linspace(max(0.01, 0.02 * S0), 2.8 * S0, 400)
    dens = risk_neutral_density(S_grid, S0, T, r, q, sigma)
    expected_ST = S0 * math.exp((r - q) * T)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=S_grid,
            y=dens,
            mode="lines",
            fill="tozeroy",
            name="Risk-neutral density",
            line=dict(width=4),
            hovertemplate="S_T=%{x:.4f}<br>density=%{y:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Heatmap(
            x=S_grid,
            y=[-0.05 * np.max(dens)],
            z=[dens],
            colorscale=COLORSCALE_MAIN,
            showscale=True,
            colorbar=dict(title="Density"),
            hoverinfo="skip",
        )
    )
    fig.add_vline(x=K, line_dash="dash", annotation_text=f"Strike={K:.2f}")
    fig.add_vline(x=expected_ST, line_dash="dot", annotation_text=f"E[S_T]={expected_ST:.2f}")
    fig.update_xaxes(title="Terminal underlying price")
    fig.update_yaxes(title="Risk-neutral density")
    return plot_layout(fig, "Continuous terminal distribution with probability heat strip", height=520)


def make_binomial_tree_figure(result: Dict[str, Any], K: float, N_display: int) -> go.Figure:
    stock_tree = result["stock_tree"]
    probability_tree = result["probability_tree"]
    option_tree = result["option_tree"]
    early_tree = result["early_exercise_tree"]
    cont_tree = result["continuation_tree"]
    intrinsic_tree = result["intrinsic_tree"]
    delta_tree = result["delta_tree"]
    bond_tree = result["bond_tree"]

    N_display = min(N_display, len(stock_tree) - 1)
    fig = go.Figure()

    edge_x: List[float] = []
    edge_y: List[float] = []
    for i in range(N_display):
        for j in range(i + 1):
            S = stock_tree[i][j]
            S_down = stock_tree[i + 1][j]
            S_up = stock_tree[i + 1][j + 1]
            edge_x += [i, i + 1, None, i, i + 1, None]
            edge_y += [S, S_down, None, S, S_up, None]
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="rgba(145,165,205,.25)"), hoverinfo="skip", name="Branches"))

    node_x: List[int] = []
    node_y: List[float] = []
    node_prob: List[float] = []
    node_value: List[float] = []
    node_cd: List[List[float]] = []
    node_text: List[str] = []

    for i in range(N_display + 1):
        for j in range(i + 1):
            S = float(stock_tree[i][j])
            V = float(option_tree[i][j])
            prob = float(probability_tree[i][j])
            cont = float(cont_tree[i][j]) if i < len(cont_tree) and len(cont_tree[i]) > j and np.isfinite(cont_tree[i][j]) else np.nan
            intrinsic = float(intrinsic_tree[i][j]) if len(intrinsic_tree[i]) > j else np.nan
            delta = float(delta_tree[i][j]) if i < len(delta_tree) and len(delta_tree[i]) > j and np.isfinite(delta_tree[i][j]) else np.nan
            bond = float(bond_tree[i][j]) if i < len(bond_tree) and len(bond_tree[i]) > j and np.isfinite(bond_tree[i][j]) else np.nan
            early = bool(early_tree[i][j]) if len(early_tree[i]) > j else False
            node_x.append(i)
            node_y.append(S)
            node_prob.append(prob)
            node_value.append(V)
            node_cd.append([i, j, S, V, prob, cont, intrinsic, delta, bond, 1.0 if early else 0.0])
            node_text.append(
                f"<b>Step</b> {i} | <b>Node</b> {j}<br>"
                f"S = {S:.6f}<br>V = {V:.6f}<br>"
                f"Risk-neutral probability = {prob:.4%}<br>"
                f"Continuation = {cont:.6f}<br>Intrinsic = {intrinsic:.6f}<br>"
                f"Delta hedge = {delta:.6f}<br>Bond = {bond:.6f}<br>Early exercise = {early}"
            )

    probs = np.array(node_prob)
    sizes = 10.0 + 55.0 * np.sqrt(probs / max(float(np.max(probs)), EPS))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            name="Nodes",
            marker=dict(
                size=sizes,
                color=node_value,
                colorscale=COLORSCALE_MAIN,
                showscale=True,
                colorbar=dict(title="Option value"),
                opacity=0.92,
                line=dict(width=1.1, color="white"),
            ),
            customdata=node_cd,
            text=node_text,
            hovertemplate="%{text}<extra></extra>",
            selected=dict(marker=dict(size=22, opacity=1.0)),
            unselected=dict(marker=dict(opacity=0.55)),
        )
    )

    ex_x = []
    ex_y = []
    for i in range(N_display + 1):
        for j in range(i + 1):
            if bool(early_tree[i][j]):
                ex_x.append(i)
                ex_y.append(stock_tree[i][j])
    if ex_x:
        fig.add_trace(go.Scatter(x=ex_x, y=ex_y, mode="markers", name="Early exercise", marker=dict(symbol="x", size=14, color="#FF426D", line=dict(width=2, color="#FF426D"))))

    fig.add_hline(y=K, line_dash="dash", annotation_text=f"Strike K={K:.2f}", annotation_position="top left")
    fig.update_xaxes(title="Time step")
    fig.update_yaxes(title="Underlying price")
    return plot_layout(fig, "Clickable binomial lattice: value, probability and early exercise", height=730)


def parse_selected_customdata(event: Any) -> Optional[List[float]]:
    try:
        points = event.selection.points  # type: ignore[attr-defined]
    except Exception:
        try:
            points = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
        except Exception:
            points = []
    if not points:
        return None
    point = points[0]
    customdata = None
    if isinstance(point, dict):
        customdata = point.get("customdata")
    else:
        customdata = getattr(point, "customdata", None)
    if customdata is None:
        return None
    if isinstance(customdata, dict):
        out = []
        for k in range(10):
            if k in customdata:
                out.append(customdata[k])
            elif str(k) in customdata:
                out.append(customdata[str(k)])
            else:
                return None
        return out
    if isinstance(customdata, np.ndarray):
        customdata = customdata.tolist()
    if isinstance(customdata, (list, tuple)) and len(customdata) >= 10:
        return list(customdata[:10])
    return None


def plotly_chart_selectable(fig: go.Figure, key: str) -> Optional[Any]:
    try:
        return st.plotly_chart(fig, use_container_width=True, key=key, on_select="rerun", selection_mode="points")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)
        return None


def make_fd_heatmap(fd: Dict[str, Any], quantity: str = "grid", title: str = "Finite-difference grid") -> go.Figure:
    Z = fd[quantity]
    fig = go.Figure(
        data=go.Heatmap(
            x=fd["tau"],
            y=fd["S"],
            z=Z.T,
            colorscale=COLORSCALE_MAIN,
            colorbar=dict(title=quantity.replace("_", " ")),
            hovertemplate="tau=%{x:.4f}<br>S=%{y:.4f}<br>value=%{z:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Time to maturity tau")
    fig.update_yaxes(title="Underlying price S")
    return plot_layout(fig, title, height=680)


def make_probability_field_figure(fd: Dict[str, Any], S0: float, r: float, q: float, sigma: float) -> go.Figure:
    S = np.asarray(fd["S"], dtype=float)
    tau = np.asarray(fd["tau"], dtype=float)

    P = probability_field(S, tau, S0, r, q, sigma)

    # Normalize each time-slice so the color scale describes where probability
    # is concentrated at that specific maturity.
    P_visual = P.copy()
    for n in range(P_visual.shape[0]):
        m = np.nanmax(P_visual[n, :])
        if m > 0:
            P_visual[n, :] = P_visual[n, :] / m

    # Do not let impossible far tails dominate the visual range.
    tau_max = max(float(np.max(tau)), 1e-8)
    drift = r - q - 0.5 * sigma ** 2

    low = S0 * math.exp(drift * tau_max - 4.0 * sigma * math.sqrt(tau_max))
    high = S0 * math.exp(drift * tau_max + 4.0 * sigma * math.sqrt(tau_max))

    low = max(float(np.min(S)), low)
    high = min(float(np.max(S)), high)

    mask = (S >= low) & (S <= high)

    if np.sum(mask) < 5:
        mask = np.ones_like(S, dtype=bool)

    S_plot = S[mask]
    P_plot = P_visual[:, mask]

    fig = go.Figure(
        data=go.Heatmap(
            x=tau,
            y=S_plot,
            z=P_plot.T,
            colorscale=COLORSCALE_PROB,
            zmin=0,
            zmax=1,
            colorbar=dict(title="relative density"),
            hovertemplate=(
                "tau=%{x:.4f}<br>"
                "S=%{y:.4f}<br>"
                "relative density=%{z:.4f}<extra></extra>"
            ),
        )
    )

    expected_path = S0 * np.exp((r - q) * tau)
    fig.add_trace(
        go.Scatter(
            x=tau,
            y=expected_path,
            mode="lines",
            line=dict(color="white", width=2, dash="dot"),
            name="E[S_tau]",
            hovertemplate="tau=%{x:.4f}<br>E[S_tau]=%{y:.4f}<extra></extra>",
        )
    )

    fig.add_hline(
        y=S0,
        line_dash="dash",
        line_width=1,
        annotation_text=f"S0={S0:.2f}",
    )

    fig.update_xaxes(title="Time to maturity tau")
    fig.update_yaxes(title="Underlying price S", range=[S_plot.min(), S_plot.max()])

    return plot_layout(
        fig,
        "Risk-neutral probability density transported through the grid",
        height=620,
    )


def make_fd_surface_figure(fd: Dict[str, Any], title: str = "Finite-difference value surface") -> go.Figure:
    # Downsample for smooth browser interaction.
    S = fd["S"]
    tau = fd["tau"]
    Z = fd["grid"]
    s_step = max(1, len(S) // 70)
    t_step = max(1, len(tau) // 70)
    S_d = S[::s_step]
    tau_d = tau[::t_step]
    Z_d = Z[::t_step, ::s_step]
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=tau_d,
            y=S_d,
            z=Z_d.T,
            colorscale=COLORSCALE_MAIN,
            opacity=0.92,
            colorbar=dict(title="V"),
            hovertemplate="tau=%{x:.4f}<br>S=%{y:.4f}<br>V=%{z:.6f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template=PLOT_TEMPLATE,
        height=720,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis_title="tau",
            yaxis_title="S",
            zaxis_title="V(S,tau)",
            camera=dict(eye=dict(x=1.6, y=1.7, z=1.05)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def make_animated_surface_cross_section(fd: Dict[str, Any], clip_quantile: float = 0.995) -> go.Figure:
    S = fd["S"]
    tau = fd["tau"]
    Z = fd["grid"]
    s_step = max(1, len(S) // 75)
    t_step = max(1, len(tau) // 65)
    S_d = S[::s_step]
    tau_d = tau[::t_step]
    Z_d = Z[::t_step, ::s_step]
    finite_abs = np.abs(Z_d[np.isfinite(Z_d)])
    clip = float(np.quantile(finite_abs, clip_quantile)) if finite_abs.size else 1.0
    clip = max(clip, 1.0)
    Z_plot = np.clip(Z_d, -clip, clip)

    k0 = 0
    line_tau = np.full_like(S_d, tau_d[k0])
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=tau_d,
            y=S_d,
            z=Z_plot.T,
            colorscale=COLORSCALE_MAIN,
            opacity=0.72,
            colorbar=dict(title="clipped V"),
            hovertemplate="tau=%{x:.4f}<br>S=%{y:.4f}<br>V=%{z:.6f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=line_tau,
            y=S_d,
            z=Z_plot[k0, :],
            mode="lines",
            name="Moving time section",
            line=dict(width=8, color="white"),
        )
    )

    frames = []
    for k in range(len(tau_d)):
        frames.append(
            go.Frame(
                data=[
                    go.Surface(x=tau_d, y=S_d, z=Z_plot.T, colorscale=COLORSCALE_MAIN, opacity=0.72, colorbar=dict(title="clipped V")),
                    go.Scatter3d(x=np.full_like(S_d, tau_d[k]), y=S_d, z=Z_plot[k, :], mode="lines", line=dict(width=8, color="white")),
                ],
                name=str(k),
            )
        )
    fig.frames = frames
    fig.update_layout(
        title="Animated 3D finite-difference surface with moving time section",
        template=PLOT_TEMPLATE,
        height=760,
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis_title="tau",
            yaxis_title="S",
            zaxis_title="V(S,tau)",
            camera=dict(eye=dict(x=1.75, y=1.55, z=1.15)),
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=1.05,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                steps=[dict(method="animate", args=[[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}], label=f"{tau_d[k]:.2f}") for k in range(len(tau_d))],
                x=0.12,
                y=0.02,
                len=0.82,
            )
        ],
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def make_error_heatmap(fd: Dict[str, Any], K: float, r: float, q: float, sigma: float, option_type: str) -> Tuple[go.Figure, np.ndarray]:
    S = fd["S"]
    tau = fd["tau"]
    err = np.zeros_like(fd["grid"])
    for n, t in enumerate(tau):
        analytic = payoff(S, K, option_type) if t <= EPS else black_scholes_price_vectorized(S, K, float(t), r, q, sigma, option_type)
        err[n, :] = fd["grid"][n, :] - analytic
    clipped = np.clip(err, -np.nanpercentile(np.abs(err), 99.0), np.nanpercentile(np.abs(err), 99.0)) if np.any(np.isfinite(err)) else err
    fig = go.Figure(
        data=go.Heatmap(
            x=tau,
            y=S,
            z=clipped.T,
            colorscale=COLORSCALE_ERROR,
            zmid=0,
            colorbar=dict(title="error"),
            hovertemplate="tau=%{x:.4f}<br>S=%{y:.4f}<br>error=%{z:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Time to maturity tau")
    fig.update_yaxes(title="Underlying price S")
    return plot_layout(fig, "Numerical error heatmap against Black-Scholes", height=620), err


def make_error_growth_figure(fd: Dict[str, Any], err: np.ndarray) -> go.Figure:
    max_err = np.nanmax(np.abs(err), axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fd["tau"], y=max_err, mode="lines+markers", name="max |error|", hovertemplate="tau=%{x:.4f}<br>max error=%{y:.6g}<extra></extra>"))
    fig.update_yaxes(title="Maximum absolute error", type="log" if np.nanmax(max_err) > 0 else None)
    fig.update_xaxes(title="Time to maturity tau")
    return plot_layout(fig, "Error growth across pseudo-time", height=470)


def make_stability_map_figure(T: float, r: float, q: float, sigma: float) -> go.Figure:
    M_values = np.arange(10, 161, 5)
    N_values = np.arange(5, 501, 10)
    Z = stability_map_data(M_values, N_values, T, r, q, sigma)
    fig = go.Figure(
        data=go.Heatmap(
            x=M_values,
            y=N_values,
            z=Z,
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="min coeff"),
            hovertemplate="M=%{x}<br>N=%{y}<br>min coefficient=%{z:.6f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Spatial grid M")
    fig.update_yaxes(title="Time grid N")
    return plot_layout(fig, "Explicit finite-difference stability map: green = non-negative coefficients", height=620)


def make_convergence_figure(S0: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str, exercise_style: str, model: str) -> go.Figure:
    Ns = np.array([5, 10, 15, 25, 40, 60, 80, 120, 160, 220, 320, 500], dtype=int)
    prices = []
    for n in Ns:
        try:
            prices.append(build_binomial_model(S0, K, T, r, q, sigma, int(n), option_type, exercise_style, model)["price"])
        except Exception:
            prices.append(np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ns, y=prices, mode="lines+markers", name="Binomial", hovertemplate="N=%{x}<br>price=%{y:.6f}<extra></extra>"))
    if exercise_style == "european":
        bs = black_scholes_merton(S0, K, T, r, q, sigma, option_type)["price"]
        fig.add_hline(y=bs, line_dash="dash", annotation_text=f"BSM={bs:.5f}", annotation_position="top right")
    fig.update_xaxes(title="Number of time steps N")
    fig.update_yaxes(title="Option value")
    return plot_layout(fig, "Binomial convergence towards the continuous-time benchmark", height=500)


def make_greek_surface(
    greek: str,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
    S0: float,
) -> go.Figure:
    S_grid = np.linspace(0.25 * S0, 2.25 * S0, 90)
    T_grid = np.linspace(0.03, T, 70)
    Z = np.zeros((len(T_grid), len(S_grid)))
    for i, tau in enumerate(T_grid):
        for j, s in enumerate(S_grid):
            Z[i, j] = black_scholes_merton(float(s), K, float(tau), r, q, sigma, option_type)[greek]
    fig = go.Figure(
        data=go.Heatmap(
            x=S_grid,
            y=T_grid,
            z=Z,
            colorscale=COLORSCALE_MAIN,
            colorbar=dict(title=greek),
            hovertemplate="S=%{x:.4f}<br>T=%{y:.4f}<br>" + greek + "=%{z:.6f}<extra></extra>",
        )
    )
    fig.add_vline(x=K, line_dash="dash", annotation_text="K")
    fig.update_xaxes(title="Underlying price S")
    fig.update_yaxes(title="Time to maturity T")
    return plot_layout(fig, f"{greek.capitalize()} surface/heatmap from Black-Scholes-Merton", height=620)
