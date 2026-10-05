"""Reusable Streamlit layout, styling, and table components."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .models import probability_field

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp {
            background:
              radial-gradient(circle at 10% 0%, rgba(0, 170, 255, 0.16), transparent 32%),
              radial-gradient(circle at 84% 8%, rgba(255, 55, 120, 0.14), transparent 30%),
              linear-gradient(180deg, #050914 0%, #070B17 44%, #02040A 100%);
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(9, 17, 34, 0.98), rgba(3, 7, 15, 0.98));
            border-right: 1px solid rgba(96, 165, 250, 0.25);
        }
        .quant-hero {
            border: 1px solid rgba(100, 181, 246, .28);
            background: linear-gradient(135deg, rgba(13, 27, 54, .92), rgba(7, 12, 26, .86));
            padding: 28px 30px;
            border-radius: 24px;
            box-shadow: 0 22px 80px rgba(0,0,0,.35);
            margin-bottom: 20px;
        }
        .quant-title {
            font-size: 2.35rem;
            font-weight: 850;
            letter-spacing: -0.04em;
            margin: 0;
            color: #F8FBFF;
        }
        .quant-subtitle {
            color: #BBD7FF;
            margin-top: 8px;
            font-size: 1rem;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, .94), rgba(6, 12, 25, .92));
            border: 1px solid rgba(96, 165, 250, .28);
            padding: 18px 20px;
            border-radius: 18px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.05), 0 12px 40px rgba(0,0,0,.25);
        }
        div[data-testid="stMetricLabel"] { color: #CFE4FF !important; }
        div[data-testid="stMetricValue"] { color: #F8FAFC !important; font-weight: 800; }
        .glass-card {
            border: 1px solid rgba(148, 163, 184, .22);
            background: rgba(15, 23, 42, .72);
            border-radius: 18px;
            padding: 18px 20px;
            margin-bottom: 14px;
        }
        .status-green { color: #40F99B; font-weight: 800; }
        .status-yellow { color: #FBBF24; font-weight: 800; }
        .status-red { color: #FB7185; font-weight: 800; }
        .small-note { color: #9FB7D6; font-size: .91rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 12px; }
        .stTabs [data-baseweb="tab"] {
            background: rgba(15, 23, 42, .88);
            border-radius: 12px 12px 0 0;
            border: 1px solid rgba(148,163,184,.18);
            padding: 10px 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="quant-hero">
            <h1 class="quant-title">Quantitative Option Pricing</h1>
            <div class="quant-subtitle">
                Quantitative finance analytics for option valuation, numerical methods, risk measures and reporting.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(items: List[Tuple[str, str, Optional[str]]], ncols: int = 4) -> None:
    cols = st.columns(ncols)
    for k, (label, value, delta) in enumerate(items):
        cols[k % ncols].metric(label, value, delta=delta)


def section_header(title: str, caption: str = "") -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def df_display(df: pd.DataFrame, height: Optional[int] = None) -> None:
    """Display a dataframe without passing height=None to Streamlit.

    Recent Streamlit versions reject height=None. This helper keeps all
    normal tables visible and numeric while still allowing fixed-height
    tables when a positive integer is provided.
    """
    if height is None:
        st.dataframe(df, use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True, height=int(height))


def make_numeric_grid_table(
    values: np.ndarray,
    row_axis: np.ndarray,
    col_axis: np.ndarray,
    row_label: str = "S",
    col_label: str = "tau",
    max_rows: int = 34,
    max_cols: int = 24,
) -> pd.DataFrame:
    """Downsample a 2D numerical grid into a readable Hull-style table.

    The returned table contains real numbers in every cell. Colors are added
    only as a visual layer through pandas Styler, never as a replacement for
    numerical values.
    """
    arr = np.asarray(values, dtype=float)
    row_axis = np.asarray(row_axis, dtype=float)
    col_axis = np.asarray(col_axis, dtype=float)

    row_idx = np.linspace(0, len(row_axis) - 1, min(max_rows, len(row_axis)), dtype=int)
    col_idx = np.linspace(0, len(col_axis) - 1, min(max_cols, len(col_axis)), dtype=int)

    # arr is expected as rows x cols.
    selected = arr[np.ix_(row_idx, col_idx)]
    table = pd.DataFrame(
        selected,
        index=[f"{row_label}={row_axis[i]:.2f}" for i in row_idx],
        columns=[f"{col_label}={col_axis[j]:.3f}" for j in col_idx],
    )
    return table


def show_colored_numeric_table(
    table: pd.DataFrame,
    precision: int = 5,
    height: int = 560,
    cmap: str = "turbo",
    caption: str | None = None,
) -> None:
    """Display a numerical table with values plus a heatmap background."""
    if caption:
        st.caption(caption)
    numeric_table = table.astype(float).replace([np.inf, -np.inf], np.nan)
    try:
        styled = numeric_table.style.format(f"{{:.{precision}f}}").background_gradient(cmap=cmap, axis=None)
        st.dataframe(styled, use_container_width=True, height=int(height))
    except Exception:
        st.dataframe(numeric_table.round(precision), use_container_width=True, height=int(height))


def finite_difference_numeric_table(fd: Dict[str, Any], quantity: str = "grid", max_rows: int = 34, max_cols: int = 24) -> pd.DataFrame:
    # fd[quantity] has shape time x space; transpose so rows are underlying prices and columns are times.
    return make_numeric_grid_table(
        np.asarray(fd[quantity]).T,
        fd["S"],
        fd["tau"],
        row_label="S",
        col_label="tau",
        max_rows=max_rows,
        max_cols=max_cols,
    )


def probability_numeric_table(fd: Dict[str, Any], S0: float, r: float, q: float, sigma: float, max_rows: int = 34, max_cols: int = 24) -> pd.DataFrame:
    P = probability_field(fd["S"], fd["tau"], S0, r, q, sigma)
    return make_numeric_grid_table(
        P.T,
        fd["S"],
        fd["tau"],
        row_label="S",
        col_label="tau",
        max_rows=max_rows,
        max_cols=max_cols,
    )
