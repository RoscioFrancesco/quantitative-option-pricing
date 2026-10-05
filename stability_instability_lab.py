"""
Stability Instability Laboratory for the explicit finite-difference Black-Scholes solver.

Run with:
    streamlit run stability_instability_lab.py

Dependencies:
    streamlit, numpy, pandas, plotly

This app is designed as a visual numerical-analysis laboratory:
- explicit finite-difference scheme for the Black-Scholes PDE;
- stability/instability diagnostics;
- heatmaps of the option value and error against Black-Scholes;
- pseudo-probability finite-difference weights alpha, beta, gamma;
- animated 3D surface with a moving time slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Page configuration and visual style
# =============================================================================

st.set_page_config(
    page_title="Finite Difference Stability Lab",
    page_icon="🧮",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #12213a 0%, #070b16 38%, #02040a 100%);
            color: #f3f7ff;
        }
        .main-header {
            padding: 1.30rem 1.55rem;
            border: 1px solid rgba(96, 165, 250, 0.28);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(17, 34, 64, 0.95), rgba(5, 12, 25, 0.90));
            box-shadow: 0 22px 45px rgba(0,0,0,0.30);
            margin-bottom: 1rem;
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.25rem;
            letter-spacing: -0.04em;
        }
        .main-header p {
            margin: 0.55rem 0 0 0;
            color: #b8d7ff;
            font-size: 1.03rem;
        }
        .metric-card {
            border: 1px solid rgba(96, 165, 250, 0.24);
            border-radius: 18px;
            padding: 1.05rem;
            background: rgba(7, 13, 28, 0.82);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            min-height: 110px;
        }
        .metric-card .label {
            color: #b9c6d8;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }
        .metric-card .value {
            color: #ffffff;
            font-size: 1.75rem;
            font-weight: 800;
            margin-top: 0.35rem;
        }
        .metric-card .caption {
            color: #93a8c6;
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }
        .stable-box {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            border: 1px solid rgba(34, 197, 94, 0.38);
            background: rgba(14, 94, 47, 0.22);
            color: #d7ffe6;
        }
        .unstable-box {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            border: 1px solid rgba(248, 113, 113, 0.45);
            background: rgba(127, 29, 29, 0.28);
            color: #ffe1e1;
        }
        .near-box {
            padding: 1rem 1.2rem;
            border-radius: 18px;
            border: 1px solid rgba(251, 191, 36, 0.45);
            background: rgba(120, 53, 15, 0.28);
            color: #fff3c4;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(96, 165, 250, 0.22);
            border-radius: 16px;
            padding: 0.75rem;
            background: rgba(8, 13, 28, 0.72);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Numerical core
# =============================================================================

OptionType = Literal["call", "put"]


@dataclass
class ExplicitFDSolution:
    S: np.ndarray
    tau: np.ndarray
    V: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    ds: float
    dt: float
    stable_by_positive_weights: bool
    min_weight: float
    max_lambda: float
    initial_price: float
    exploded_step: int | None


_NORMAL = NormalDist()


def normal_cdf_scalar(x: float) -> float:
    return _NORMAL.cdf(float(x))


def normal_cdf_array(x: np.ndarray) -> np.ndarray:
    # statistics.NormalDist is in the Python standard library. np.vectorize is fine here
    # because the grids are modest and the goal is robust portability without scipy.
    return np.vectorize(normal_cdf_scalar, otypes=[float])(x)


def black_scholes_price_array(
    S: np.ndarray,
    K: float,
    tau: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
) -> np.ndarray:
    S = np.asarray(S, dtype=float)

    if tau <= 0 or sigma <= 0:
        if option_type == "call":
            return np.maximum(S - K, 0.0)
        return np.maximum(K - S, 0.0)

    safe_S = np.maximum(S, 1.0e-12)
    vol_sqrt = sigma * math.sqrt(tau)
    d1 = (np.log(safe_S / K) + (r - q + 0.5 * sigma * sigma) * tau) / vol_sqrt
    d2 = d1 - vol_sqrt

    if option_type == "call":
        price = safe_S * math.exp(-q * tau) * normal_cdf_array(d1) - K * math.exp(-r * tau) * normal_cdf_array(d2)
        price = np.where(S <= 1.0e-12, 0.0, price)
    else:
        price = K * math.exp(-r * tau) * normal_cdf_array(-d2) - safe_S * math.exp(-q * tau) * normal_cdf_array(-d1)
        price = np.where(S <= 1.0e-12, K * math.exp(-r * tau), price)

    return price


def black_scholes_price_scalar(
    S: float,
    K: float,
    tau: float,
    r: float,
    q: float,
    sigma: float,
    option_type: OptionType,
) -> float:
    return float(black_scholes_price_array(np.array([S]), K, tau, r, q, sigma, option_type)[0])


def payoff(S: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def boundary_value(Smax: float, tau: float, K: float, r: float, q: float, option_type: OptionType) -> tuple[float, float]:
    if option_type == "call":
        lower = 0.0
        upper = max(Smax * math.exp(-q * tau) - K * math.exp(-r * tau), 0.0)
    else:
        lower = K * math.exp(-r * tau)
        upper = 0.0
    return lower, upper


def explicit_fd_solver(
    Smax: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    M: int,
    N: int,
    option_type: OptionType,
    explosion_cap: float = 1.0e8,
) -> ExplicitFDSolution:
    """Solve the Black-Scholes PDE in time-to-maturity tau by an explicit FD scheme.

    Grid:
        S_i = i * dS, i = 0,...,M
        tau_n = n * dt, n = 0,...,N

    tau = 0 is maturity, V(S,0)=payoff(S).
    The required option price today is V(S0,T), obtained by interpolation outside this function.
    """
    if M < 4:
        raise ValueError("M must be at least 4.")
    if N < 2:
        raise ValueError("N must be at least 2.")
    if Smax <= 0 or T <= 0 or K <= 0 or sigma <= 0:
        raise ValueError("Smax, K, T and sigma must be positive.")

    ds = Smax / M
    dt = T / N
    S = np.linspace(0.0, Smax, M + 1)
    tau = np.linspace(0.0, T, N + 1)
    V = np.zeros((M + 1, N + 1), dtype=float)

    V[:, 0] = payoff(S, K, option_type)
    lower, upper = boundary_value(Smax, 0.0, K, r, q, option_type)
    V[0, 0] = lower
    V[M, 0] = upper

    i = np.arange(1, M, dtype=float)
    drift = r - q

    alpha = 0.5 * dt * (sigma * sigma * i * i - drift * i)
    beta = 1.0 - dt * (sigma * sigma * i * i + r)
    gamma = 0.5 * dt * (sigma * sigma * i * i + drift * i)

    min_weight = float(np.min(np.concatenate([alpha, beta, gamma])))
    stable_by_positive_weights = bool(np.all(alpha >= -1.0e-14) and np.all(beta >= -1.0e-14) and np.all(gamma >= -1.0e-14))
    max_lambda = float(np.max((sigma * sigma * (np.arange(1, M, dtype=float) ** 2) + r) * dt))

    exploded_step: int | None = None

    for n in range(0, N):
        next_tau = tau[n + 1]
        lower, upper = boundary_value(Smax, next_tau, K, r, q, option_type)
        V[0, n + 1] = lower
        V[M, n + 1] = upper

        V[1:M, n + 1] = (
            alpha * V[0:M - 1, n]
            + beta * V[1:M, n]
            + gamma * V[2:M + 1, n]
        )

        if not np.all(np.isfinite(V[:, n + 1])):
            V[:, n + 1] = np.nan_to_num(V[:, n + 1], nan=0.0, posinf=explosion_cap, neginf=-explosion_cap)

        max_abs = float(np.max(np.abs(V[:, n + 1])))
        if max_abs > explosion_cap and exploded_step is None:
            exploded_step = n + 1

        # Safety clipping: this preserves the visual idea of explosion without letting
        # floating point overflow destroy the interactive session.
        V[:, n + 1] = np.clip(V[:, n + 1], -explosion_cap, explosion_cap)

    return ExplicitFDSolution(
        S=S,
        tau=tau,
        V=V,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        ds=ds,
        dt=dt,
        stable_by_positive_weights=stable_by_positive_weights,
        min_weight=min_weight,
        max_lambda=max_lambda,
        initial_price=float("nan"),
        exploded_step=exploded_step,
    )


def interpolate_price(solution: ExplicitFDSolution, S0: float) -> float:
    return float(np.interp(S0, solution.S, solution.V[:, -1]))


def signed_log_transform(Z: np.ndarray) -> np.ndarray:
    return np.sign(Z) * np.log10(1.0 + np.abs(Z))


def robust_clip_for_display(Z: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    finite = Z[np.isfinite(Z)]
    if finite.size == 0:
        return np.zeros_like(Z)
    lo = np.percentile(finite, low_q)
    hi = np.percentile(finite, high_q)
    if math.isclose(float(lo), float(hi)):
        return Z
    return np.clip(Z, lo, hi)


def bs_surface(solution: ExplicitFDSolution, K: float, r: float, q: float, sigma: float, option_type: OptionType) -> np.ndarray:
    B = np.zeros_like(solution.V)
    for n, tau_n in enumerate(solution.tau):
        B[:, n] = black_scholes_price_array(solution.S, K, float(tau_n), r, q, sigma, option_type)
    return B


def compute_greeks_on_grid(solution: ExplicitFDSolution) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = solution.V
    ds = solution.ds
    dt = solution.dt
    delta = np.full_like(V, np.nan)
    gamma = np.full_like(V, np.nan)
    theta_tau = np.full_like(V, np.nan)

    delta[1:-1, :] = (V[2:, :] - V[:-2, :]) / (2.0 * ds)
    gamma[1:-1, :] = (V[2:, :] - 2.0 * V[1:-1, :] + V[:-2, :]) / (ds * ds)
    theta_tau[:, 1:] = (V[:, 1:] - V[:, :-1]) / dt

    # Financial theta is dV/dt_calendar = -dV/dtau.
    theta = -theta_tau
    return delta, gamma, theta


# =============================================================================
# Plotting functions
# =============================================================================

PLOT_TEMPLATE = "plotly_dark"
HEATMAP_SCALE = "Turbo"
ERROR_SCALE = "Inferno"
STABILITY_SCALE = [[0.0, "#16a34a"], [0.49, "#22c55e"], [0.50, "#f59e0b"], [0.72, "#f97316"], [1.0, "#dc2626"]]


def make_solution_heatmap(solution: ExplicitFDSolution, display_mode: str) -> go.Figure:
    if display_mode == "Signed log explosion scale":
        Z = signed_log_transform(solution.V)
        color_title = "sign(V) log10(1+|V|)"
    else:
        Z = robust_clip_for_display(solution.V)
        color_title = "Option value"

    fig = go.Figure(
        data=go.Heatmap(
            x=solution.tau,
            y=solution.S,
            z=Z,
            colorscale=HEATMAP_SCALE,
            colorbar=dict(title=color_title),
            hovertemplate=(
                "Time to maturity τ: %{x:.5f}<br>"
                "Underlying S: %{y:.4f}<br>"
                "Displayed value: %{z:.5f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Finite-difference solution heatmap: V(S, τ)",
        xaxis_title="Time to maturity τ",
        yaxis_title="Underlying price S",
        height=620,
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig


def make_error_heatmap(solution: ExplicitFDSolution, B: np.ndarray) -> go.Figure:
    error = np.abs(solution.V - B)
    Z = np.log10(1.0 + error)

    fig = go.Figure(
        data=go.Heatmap(
            x=solution.tau,
            y=solution.S,
            z=Z,
            colorscale=ERROR_SCALE,
            colorbar=dict(title="log10(1+abs error))"),
            hovertemplate=(
                "Time to maturity τ: %{x:.5f}<br>"
                "Underlying S: %{y:.4f}<br>"
                "log error: %{z:.5f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Error heatmap against Black-Scholes closed form",
        xaxis_title="Time to maturity τ",
        yaxis_title="Underlying price S",
        height=620,
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig


def make_error_growth_figure(solution: ExplicitFDSolution, B: np.ndarray) -> go.Figure:
    abs_error = np.abs(solution.V - B)
    max_error = np.nanmax(abs_error, axis=0)
    rms_error = np.sqrt(np.nanmean(abs_error ** 2, axis=0))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=solution.tau,
            y=max_error,
            mode="lines",
            name="Max error",
            hovertemplate="τ: %{x:.5f}<br>Max error: %{y:.6g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=solution.tau,
            y=rms_error,
            mode="lines",
            name="RMS error",
            hovertemplate="τ: %{x:.5f}<br>RMS error: %{y:.6g}<extra></extra>",
        )
    )
    fig.update_yaxes(type="log")
    fig.update_layout(
        title="Growth of numerical error over pseudo-time τ",
        xaxis_title="Time to maturity τ",
        yaxis_title="Error, log scale",
        height=460,
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig


def make_weights_figure(solution: ExplicitFDSolution) -> go.Figure:
    idx = np.arange(1, len(solution.S) - 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=solution.alpha, mode="lines", name="αᵢ: down weight"))
    fig.add_trace(go.Scatter(x=idx, y=solution.beta, mode="lines", name="βᵢ: middle weight"))
    fig.add_trace(go.Scatter(x=idx, y=solution.gamma, mode="lines", name="γᵢ: up weight"))
    fig.add_hline(y=0.0, line_dash="dash", line_width=1, annotation_text="zero weight")
    fig.add_hline(y=1.0, line_dash="dot", line_width=1, annotation_text="unit weight")
    fig.update_layout(
        title="Explicit-scheme pseudo-probability weights",
        xaxis_title="Spatial index i",
        yaxis_title="Weight",
        height=500,
        template=PLOT_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig


def make_stability_map(
    T: float,
    r: float,
    q: float,
    sigma: float,
    M_current: int,
    N_current: int,
) -> go.Figure:
    M_values = np.arange(10, 181, 2)
    N_values = np.arange(10, 801, 10)
    Z = np.zeros((len(N_values), len(M_values)))

    for row, N in enumerate(N_values):
        dt = T / N
        for col, M in enumerate(M_values):
            i = np.arange(1, M, dtype=float)
            drift = r - q
            alpha = 0.5 * dt * (sigma * sigma * i * i - drift * i)
            beta = 1.0 - dt * (sigma * sigma * i * i + r)
            gamma = 0.5 * dt * (sigma * sigma * i * i + drift * i)
            min_weight = np.min(np.concatenate([alpha, beta, gamma]))
            # 0 = stable, 1 = unstable, with a transition band for visual readability.
            if min_weight >= 0:
                Z[row, col] = 0.0
            elif min_weight > -0.05:
                Z[row, col] = 0.5
            else:
                Z[row, col] = 1.0

    fig = go.Figure(
        data=go.Heatmap(
            x=M_values,
            y=N_values,
            z=Z,
            colorscale=STABILITY_SCALE,
            zmin=0,
            zmax=1,
            colorbar=dict(
                title="Regime",
                tickvals=[0, 0.5, 1],
                ticktext=["stable", "near", "unstable"],
            ),
            hovertemplate="M: %{x}<br>N: %{y}<br>regime score: %{z}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[M_current],
            y=[N_current],
            mode="markers",
            marker=dict(size=16, symbol="x", line=dict(width=3), color="white"),
            name="current grid",
            hovertemplate="Current grid<br>M: %{x}<br>N: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Stability map: spatial steps M vs time steps N",
        xaxis_title="Spatial steps M",
        yaxis_title="Time steps N",
        height=560,
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig


def make_cross_section_figure(
    solution: ExplicitFDSolution,
    B: np.ndarray,
    tau_index: int,
    S0: float,
    K: float,
) -> go.Figure:
    tau_value = solution.tau[tau_index]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=solution.S,
            y=solution.V[:, tau_index],
            mode="lines",
            name="Explicit FD",
            hovertemplate="S: %{x:.4f}<br>FD: %{y:.6g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=solution.S,
            y=B[:, tau_index],
            mode="lines",
            name="Black-Scholes",
            line=dict(dash="dash"),
            hovertemplate="S: %{x:.4f}<br>BS: %{y:.6g}<extra></extra>",
        )
    )
    fig.add_vline(x=S0, line_dash="dot", annotation_text=f"S0={S0:.2f}")
    fig.add_vline(x=K, line_dash="dash", annotation_text=f"K={K:.2f}")
    fig.update_layout(
        title=f"Cross-section V(S, τ={tau_value:.4f})",
        xaxis_title="Underlying price S",
        yaxis_title="Option value",
        height=520,
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig


def make_probability_density_figure(
    S_grid: np.ndarray,
    S0: float,
    tau: float,
    r: float,
    q: float,
    sigma: float,
    K: float,
) -> go.Figure:
    safe_S = np.maximum(S_grid, 1.0e-12)
    if tau <= 0:
        density = np.zeros_like(S_grid)
        density[np.argmin(np.abs(S_grid - S0))] = 1.0
    else:
        mu = math.log(max(S0, 1.0e-12)) + (r - q - 0.5 * sigma * sigma) * tau
        denom = safe_S * sigma * math.sqrt(2 * math.pi * tau)
        density = np.exp(-((np.log(safe_S) - mu) ** 2) / (2.0 * sigma * sigma * tau)) / denom
        density[0] = 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=S_grid,
            y=density,
            mode="lines",
            name="Risk-neutral density",
            fill="tozeroy",
            hovertemplate="S: %{x:.4f}<br>density: %{y:.6g}<extra></extra>",
        )
    )
    fig.add_vline(x=K, line_dash="dash", annotation_text=f"K={K:.2f}")
    fig.add_vline(x=S0, line_dash="dot", annotation_text=f"S0={S0:.2f}")
    fig.update_layout(
        title=f"Risk-neutral probability density at τ={tau:.4f}",
        xaxis_title="Underlying price S",
        yaxis_title="Density",
        height=430,
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig


def make_animated_3d_surface(solution: ExplicitFDSolution, display_mode: str, max_frames: int = 45) -> go.Figure:
    # Downsample for browser performance.
    max_s_points = 90
    max_t_points = 90
    s_stride = max(1, int(math.ceil(len(solution.S) / max_s_points)))
    t_stride = max(1, int(math.ceil(len(solution.tau) / max_t_points)))

    S_plot = solution.S[::s_stride]
    tau_plot = solution.tau[::t_stride]
    V_plot = solution.V[::s_stride, ::t_stride]

    if display_mode == "Signed log explosion scale":
        Z = signed_log_transform(V_plot)
        z_title = "sign(V) log10(1+|V|)"
    else:
        Z = robust_clip_for_display(V_plot)
        z_title = "Option value"

    # Surface expects z as rows over y and columns over x.
    surface = go.Surface(
        x=tau_plot,
        y=S_plot,
        z=Z,
        colorscale=HEATMAP_SCALE,
        opacity=0.86,
        colorbar=dict(title=z_title),
        name="V(S,τ)",
        hovertemplate="τ: %{x:.5f}<br>S: %{y:.4f}<br>Z: %{z:.5f}<extra></extra>",
    )

    frame_positions = np.linspace(0, len(tau_plot) - 1, min(max_frames, len(tau_plot)), dtype=int)
    frame_positions = np.unique(frame_positions)
    first_idx = int(frame_positions[0])

    moving_slice = go.Scatter3d(
        x=np.full_like(S_plot, tau_plot[first_idx], dtype=float),
        y=S_plot,
        z=Z[:, first_idx],
        mode="lines",
        line=dict(width=7, color="#ffffff"),
        name="moving time slice",
        hovertemplate="Moving slice<br>τ: %{x:.5f}<br>S: %{y:.4f}<br>Z: %{z:.5f}<extra></extra>",
    )

    frames = []
    for idx in frame_positions:
        idx = int(idx)
        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=np.full_like(S_plot, tau_plot[idx], dtype=float),
                        y=S_plot,
                        z=Z[:, idx],
                        mode="lines",
                        line=dict(width=7, color="#ffffff"),
                    )
                ],
                traces=[1],
                name=f"{idx}",
            )
        )

    fig = go.Figure(data=[surface, moving_slice], frames=frames)

    slider_steps = [
        {
            "args": [
                [f"{int(idx)}"],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{tau_plot[int(idx)]:.2f}",
            "method": "animate",
        }
        for idx in frame_positions
    ]

    fig.update_layout(
        title="Animated 3D finite-difference surface: moving time slice",
        template=PLOT_TEMPLATE,
        height=740,
        scene=dict(
            xaxis_title="Time to maturity τ",
            yaxis_title="Underlying price S",
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.55, y=-1.75, z=1.12)),
        ),
        margin=dict(l=0, r=0, t=70, b=0),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": 0.02,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 110, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "τ = "},
                "pad": {"t": 55},
                "steps": slider_steps,
            }
        ],
    )
    return fig


def make_probability_color_strip(solution: ExplicitFDSolution, S0: float, r: float, q: float, sigma: float) -> go.Figure:
    density_matrix = np.zeros_like(solution.V)
    for n, tau_n in enumerate(solution.tau):
        safe_S = np.maximum(solution.S, 1.0e-12)
        if tau_n <= 0:
            density = np.zeros_like(solution.S)
            density[np.argmin(np.abs(solution.S - S0))] = 1.0
        else:
            mu = math.log(max(S0, 1.0e-12)) + (r - q - 0.5 * sigma * sigma) * tau_n
            denom = safe_S * sigma * math.sqrt(2 * math.pi * tau_n)
            density = np.exp(-((np.log(safe_S) - mu) ** 2) / (2.0 * sigma * sigma * tau_n)) / denom
            density[0] = 0.0
        density_matrix[:, n] = density

    fig = go.Figure(
        data=go.Heatmap(
            x=solution.tau,
            y=solution.S,
            z=density_matrix,
            colorscale=HEATMAP_SCALE,
            colorbar=dict(title="density"),
            hovertemplate="τ: %{x:.5f}<br>S: %{y:.4f}<br>density: %{z:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Risk-neutral probability density color field",
        xaxis_title="Time to maturity τ",
        yaxis_title="Underlying price S",
        height=560,
        template=PLOT_TEMPLATE,
        margin=dict(l=20, r=20, t=70, b=40),
    )
    return fig


# =============================================================================
# Streamlit user interface
# =============================================================================

st.markdown(
    """
    <div class="main-header">
        <h1>Finite Difference Stability Laboratory</h1>
        <p>
            Interactive numerical-analysis module for the Black-Scholes PDE: explicit finite differences,
            pseudo-probability weights, stability breakdown, error growth, and animated 3D solution surfaces.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model parameters")

    preset = st.radio(
        "Numerical regime preset",
        ["Stable demonstration", "Unstable demonstration", "Custom"],
        index=1,
    )

    S0 = st.number_input("Initial underlying S0", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    K = st.number_input("Strike K", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
    T = st.slider("Maturity T", min_value=0.05, max_value=5.0, value=1.0, step=0.05)
    r = st.slider("Risk-free rate r", min_value=-0.05, max_value=0.25, value=0.05, step=0.005, format="%.3f")
    q = st.slider("Dividend yield q", min_value=0.0, max_value=0.15, value=0.00, step=0.005, format="%.3f")
    sigma = st.slider("Volatility σ", min_value=0.05, max_value=1.50, value=0.35, step=0.01, format="%.2f")
    option_type: OptionType = st.radio("Option type", ["call", "put"], horizontal=True)  # type: ignore[assignment]

    st.divider()
    st.header("Finite-difference grid")

    default_Smax_mult = 3.0
    default_M = 90
    default_N = 70
    if preset == "Stable demonstration":
        default_M = 60
        default_N = 800
        default_Smax_mult = 3.0
    elif preset == "Unstable demonstration":
        default_M = 95
        default_N = 60
        default_Smax_mult = 3.0

    Smax_mult = st.slider("Smax multiplier", min_value=1.5, max_value=6.0, value=default_Smax_mult, step=0.1)
    Smax = Smax_mult * max(S0, K)
    M = st.slider("Spatial steps M", min_value=20, max_value=180, value=default_M, step=1)
    N = st.slider("Time steps N", min_value=10, max_value=1200, value=default_N, step=10)

    display_mode = st.selectbox(
        "Display scale",
        ["Signed log explosion scale", "Option value scale"],
        index=0,
        help="The signed log scale makes unstable explosions readable without flattening the rest of the surface.",
    )

    st.caption("For the explicit scheme, instability is easy to trigger with large M, small N, or large volatility.")


try:
    solution = explicit_fd_solver(
        Smax=Smax,
        K=K,
        T=T,
        r=r,
        q=q,
        sigma=sigma,
        M=M,
        N=N,
        option_type=option_type,
    )
    solution.initial_price = interpolate_price(solution, S0)
    bs_today = black_scholes_price_scalar(S0, K, T, r, q, sigma, option_type)
    error_today = abs(solution.initial_price - bs_today)
    B = bs_surface(solution, K, r, q, sigma, option_type)
    delta_grid, gamma_grid, theta_grid = compute_greeks_on_grid(solution)

    status = "Stable" if solution.stable_by_positive_weights else "Unstable"
    if solution.stable_by_positive_weights:
        status_html = "<div class='stable-box'><b>Stable regime detected.</b><br>The explicit stencil weights behave like non-negative pseudo-probabilities.</div>"
    elif solution.min_weight > -0.05:
        status_html = "<div class='near-box'><b>Near instability.</b><br>At least one pseudo-probability weight is slightly negative. Oscillations may appear.</div>"
    else:
        status_html = "<div class='unstable-box'><b>Unstable regime detected.</b><br>The explicit scheme contains negative or explosive weights. Numerical oscillations are expected.</div>"

    st.markdown(status_html, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"<div class='metric-card'><div class='label'>FD price</div><div class='value'>{solution.initial_price:.5g}</div><div class='caption'>Interpolated at S0</div></div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Black-Scholes</div><div class='value'>{bs_today:.5g}</div><div class='caption'>Closed-form benchmark</div></div>",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Absolute error</div><div class='value'>{error_today:.4g}</div><div class='caption'>At S0, τ=T</div></div>",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"<div class='metric-card'><div class='label'>min weight</div><div class='value'>{solution.min_weight:.4g}</div><div class='caption'>min αᵢ, βᵢ, γᵢ</div></div>",
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"<div class='metric-card'><div class='label'>max λ</div><div class='value'>{solution.max_lambda:.4g}</div><div class='caption'>max (σ²i²+r)Δt</div></div>",
            unsafe_allow_html=True,
        )

    if solution.exploded_step is not None:
        st.warning(
            f"Explosion cap reached at pseudo-time step n={solution.exploded_step}. "
            "Values are clipped for visualization so the browser does not crash."
        )

    tabs = st.tabs(
        [
            "Theory",
            "Solution heatmap",
            "Animated 3D surface",
            "Error diagnostics",
            "Stability map",
            "Pseudo-probabilities",
            "Cell inspector",
            "Data table",
        ]
    )

    with tabs[0]:
        st.subheader("Black-Scholes PDE and explicit finite differences")
        st.markdown(
            "The Black-Scholes-Merton PDE for a derivative value V(S,t) is transformed using time-to-maturity "
            "τ = T - t. The forward equation in τ is solved from the payoff at maturity toward today."
        )
        st.latex(r"\frac{\partial V}{\partial \tau}=\frac{1}{2}\sigma^2S^2\frac{\partial^2V}{\partial S^2}+(r-q)S\frac{\partial V}{\partial S}-rV")
        st.markdown("On the grid S_i=iΔS and τ_n=nΔτ, the explicit scheme is")
        st.latex(r"V_i^{n+1}=\alpha_iV_{i-1}^{n}+\beta_iV_i^{n}+\gamma_iV_{i+1}^{n}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.latex(r"\alpha_i=\frac{\Delta\tau}{2}\left(\sigma^2i^2-(r-q)i\right)")
        with c2:
            st.latex(r"\beta_i=1-\Delta\tau\left(\sigma^2i^2+r\right)")
        with c3:
            st.latex(r"\gamma_i=\frac{\Delta\tau}{2}\left(\sigma^2i^2+(r-q)i\right)")
        st.markdown(
            "If these coefficients behave like non-negative probabilities, the grid is usually well behaved. "
            "When one or more weights become negative, the method can create artificial oscillations that grow over time."
        )
        st.latex(r"\alpha_i\ge 0,\qquad \beta_i\ge 0,\qquad \gamma_i\ge 0")
        st.markdown(
            "This is not a financial phenomenon. It is a numerical artifact: the finite-difference scheme no longer "
            "approximates the continuous PDE in a stable way."
        )

    with tabs[1]:
        left, right = st.columns([1.15, 0.85])
        with left:
            st.plotly_chart(make_solution_heatmap(solution, display_mode), use_container_width=True)
        with right:
            tau_index = st.slider("Cross-section τ index", 0, N, min(N, max(1, N // 2)))
            st.plotly_chart(make_cross_section_figure(solution, B, tau_index, S0, K), use_container_width=True)
            st.plotly_chart(make_probability_density_figure(solution.S, S0, float(solution.tau[tau_index]), r, q, sigma, K), use_container_width=True)

        st.plotly_chart(make_probability_color_strip(solution, S0, r, q, sigma), use_container_width=True)

    with tabs[2]:
        st.info("Press Play. The white curve is a moving time slice crossing the 3D finite-difference surface.")
        st.plotly_chart(make_animated_3d_surface(solution, display_mode), use_container_width=True)

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(make_error_heatmap(solution, B), use_container_width=True)
        with c2:
            st.plotly_chart(make_error_growth_figure(solution, B), use_container_width=True)

    with tabs[4]:
        st.markdown(
            "The point marked with a white cross is the current grid. Green areas satisfy the positive-weight condition; "
            "red areas violate it and usually produce unstable behavior."
        )
        st.plotly_chart(make_stability_map(T, r, q, sigma, M, N), use_container_width=True)

    with tabs[5]:
        st.plotly_chart(make_weights_figure(solution), use_container_width=True)
        weights_df = pd.DataFrame(
            {
                "i": np.arange(1, M),
                "S_i": solution.S[1:M],
                "alpha_down": solution.alpha,
                "beta_middle": solution.beta,
                "gamma_up": solution.gamma,
                "sum": solution.alpha + solution.beta + solution.gamma,
                "negative_weight": (solution.alpha < 0) | (solution.beta < 0) | (solution.gamma < 0),
            }
        )
        st.dataframe(weights_df, use_container_width=True, height=380)

    with tabs[6]:
        st.subheader("Finite-difference cell inspector")
        st.markdown("Select a grid cell and inspect the local numerical stencil.")
        ci1, ci2, ci3 = st.columns(3)
        with ci1:
            i_sel = st.slider("Spatial index i", 1, M - 1, min(M - 1, max(1, int(round(S0 / solution.ds)))))
        with ci2:
            n_sel = st.slider("Time index n", 0, N, N)
        with ci3:
            tau_sel = float(solution.tau[n_sel])
            S_sel = float(solution.S[i_sel])
            st.metric("Selected point", f"S={S_sel:.2f}, τ={tau_sel:.3f}")

        value = float(solution.V[i_sel, n_sel])
        benchmark = float(B[i_sel, n_sel])
        delta_val = float(delta_grid[i_sel, n_sel]) if np.isfinite(delta_grid[i_sel, n_sel]) else float("nan")
        gamma_val = float(gamma_grid[i_sel, n_sel]) if np.isfinite(gamma_grid[i_sel, n_sel]) else float("nan")
        theta_val = float(theta_grid[i_sel, n_sel]) if np.isfinite(theta_grid[i_sel, n_sel]) else float("nan")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("FD value", f"{value:.6g}")
        m2.metric("BS value", f"{benchmark:.6g}")
        m3.metric("Delta", f"{delta_val:.6g}")
        m4.metric("Gamma", f"{gamma_val:.6g}")
        m5.metric("Theta", f"{theta_val:.6g}")

        local_idx = i_sel - 1
        stencil_df = pd.DataFrame(
            {
                "coefficient": ["alpha_i", "beta_i", "gamma_i"],
                "target node": ["V[i-1,n]", "V[i,n]", "V[i+1,n]"],
                "value": [
                    solution.alpha[local_idx],
                    solution.beta[local_idx],
                    solution.gamma[local_idx],
                ],
                "interpretation": [
                    "down pseudo-probability weight",
                    "middle pseudo-probability weight",
                    "up pseudo-probability weight",
                ],
            }
        )
        st.dataframe(stencil_df, use_container_width=True, height=180)

    with tabs[7]:
        st.subheader("Numerical table V(S_i, τ_n)")
        st.caption("Rows are spatial nodes S_i. Columns are time-to-maturity levels τ_n. The table is downsampled if the grid is large.")
        max_cols = 26
        col_indices = np.linspace(0, N, min(max_cols, N + 1), dtype=int)
        table = pd.DataFrame({"S_i": solution.S})
        for idx in col_indices:
            table[f"tau={solution.tau[idx]:.3f}"] = solution.V[:, idx]
        st.dataframe(table, use_container_width=True, height=520)

        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download displayed table as CSV",
            data=csv,
            file_name="finite_difference_instability_table.csv",
            mime="text/csv",
        )

except Exception as exc:
    st.error(f"The numerical laboratory could not be computed: {exc}")
    st.info("Try lowering M, increasing N, or reducing volatility.")
