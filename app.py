"""Streamlit entry point for Quantitative Option Pricing."""

from __future__ import annotations

import numpy as np
import streamlit as st

from option_pricing.components import hero, inject_css
from option_pricing.models import (
    black_scholes_merton, build_binomial_model, finite_difference_solver,
)
from option_pricing.pages import (
    render_binomial_lab, render_dashboard, render_distribution_lab, render_fd_lab,
    render_greeks_lab, render_model_comparison, render_monte_carlo_lab,
    render_report_center, render_stability_lab, render_theory,
)


def main() -> None:
    st.set_page_config(page_title="Quantitative Option Pricing", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
    inject_css()
    hero()

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Section",
            [
                "Executive Dashboard",
                "Model Comparison",
                "Distribution Lab",
                "Binomial Tree Lab",
                "Finite Difference Lab",
                "Stability Lab",
                "Monte Carlo Lab",
                "Greeks Lab",
                "Theory",
                "Report Center",
            ],
        )

        st.divider()
        st.markdown("## Contract")
        option_type = st.radio("Option type", ["call", "put"], horizontal=True)
        exercise_style = st.radio("Exercise style", ["european", "american"], horizontal=True)
        S0 = st.slider("Initial price S0", 10.0, 400.0, 100.0, 1.0)
        K = st.slider("Strike K", 10.0, 400.0, 100.0, 1.0)
        T = st.slider("Maturity T", 0.05, 5.0, 1.0, 0.05)
        r = st.slider("Risk-free rate r", -0.05, 0.25, 0.05, 0.0025, format="%.4f")
        q = st.slider("Dividend yield q", 0.0, 0.15, 0.00, 0.0025, format="%.4f")
        sigma = st.slider("Volatility sigma", 0.03, 1.25, 0.20, 0.01)

        st.divider()
        st.markdown("## Numerical engines")
        binomial_model = st.selectbox("Binomial model", ["CRR", "Jarrow-Rudd"])
        N_binomial = st.slider("Binomial steps", 5, 700, 180, 5)
        fd_method = st.selectbox("Finite-difference method", ["Crank-Nicolson", "Implicit", "Explicit"], index=0)
        Smax_mult = st.slider("Smax / max(S0,K)", 1.4, 5.0, 3.0, 0.1)
        M_fd = st.slider("FD spatial grid M", 20, 180, 90, 5)
        N_fd = st.slider("FD time grid N", 20, 600, 180, 10)
        omega = st.slider("PSOR omega", 1.0, 1.8, 1.25, 0.05)

    Smax = float(Smax_mult * max(S0, K))
    params = {
        "S0": float(S0),
        "K": float(K),
        "T": float(T),
        "r": float(r),
        "q": float(q),
        "sigma": float(sigma),
        "option_type": option_type,
        "exercise_style": exercise_style,
        "binomial_model": binomial_model,
        "N_binomial": int(N_binomial),
        "fd_method": fd_method,
        "Smax": float(Smax),
        "M_fd": int(M_fd),
        "N_fd": int(N_fd),
        "omega": float(omega),
    }

    bsm = black_scholes_merton(S0, K, T, r, q, sigma, option_type)

    try:
        binom = build_binomial_model(S0, K, T, r, q, sigma, N_binomial, option_type, exercise_style, binomial_model)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    fd = finite_difference_solver(S0, K, T, r, q, sigma, Smax, M_fd, N_fd, option_type, exercise_style, fd_method, omega)

    # Compact status strip
    price_spread = fd["price"] - bsm["price"] if exercise_style == "european" else np.nan
    st.markdown(
        f"""
        <div class="glass-card">
        <b>Current contract:</b> {exercise_style.capitalize()} {option_type.upper()} | S0={S0:.2f}, K={K:.2f}, T={T:.2f}, r={r:.2%}, q={q:.2%}, σ={sigma:.2%} &nbsp; | &nbsp;
        <b>BSM European:</b> {bsm['price']:.5f} &nbsp; <b>Binomial:</b> {binom['price']:.5f} &nbsp; <b>FD:</b> {fd['price']:.5f}
        {f"&nbsp; <b>FD-BSM:</b> {price_spread:+.5f}" if np.isfinite(price_spread) else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Executive Dashboard":
        render_dashboard(params, bsm, binom, fd)
    elif page == "Model Comparison":
        render_model_comparison(params, bsm, binom, fd)
    elif page == "Distribution Lab":
        render_distribution_lab(params, binom, fd)
    elif page == "Binomial Tree Lab":
        render_binomial_lab(params, binom)
    elif page == "Finite Difference Lab":
        render_fd_lab(params, fd, bsm)
    elif page == "Stability Lab":
        render_stability_lab(params)
    elif page == "Monte Carlo Lab":
        render_monte_carlo_lab(params, bsm)
    elif page == "Greeks Lab":
        render_greeks_lab(params, bsm, fd)
    elif page == "Theory":
        render_theory(params)
    elif page == "Report Center":
        render_report_center(params, bsm, binom, fd)


if __name__ == "__main__":
    main()
