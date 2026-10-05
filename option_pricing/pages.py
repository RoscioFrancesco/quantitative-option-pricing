"""Streamlit page renderers, grouped by user-facing laboratory section."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .charts import (
    COLORSCALE_PROB,
    make_animated_surface_cross_section, make_binomial_tree_figure,
    make_convergence_figure, make_error_growth_figure, make_error_heatmap,
    make_fd_heatmap, make_fd_surface_figure, make_greek_surface,
    make_lognormal_distribution_figure, make_price_comparison_figure,
    make_probability_field_figure, make_stability_map_figure,
    make_payoff_value_figure, parse_selected_customdata, plot_layout,
    plotly_chart_selectable,
)
from .components import (
    df_display, finite_difference_numeric_table, make_numeric_grid_table, metric_grid,
    probability_numeric_table, section_header, show_colored_numeric_table,
)
from .models import (
    black_scholes_merton, build_binomial_model, explicit_stability_coefficients,
    finite_difference_solver, monte_carlo_european, payoff, probability_field,
    risk_neutral_density, simulate_gbm_paths,
)
from .reports import make_markdown_report, minimal_pdf_bytes


def render_dashboard(params: Dict[str, Any], bsm: Dict[str, float], binom: Dict[str, Any], fd: Dict[str, Any]) -> None:
    section_header("Executive Dashboard", "European analytic benchmark alongside lattice and PDE prices for the selected exercise style.")
    price_diff_binom = binom["price"] - bsm["price"] if params["exercise_style"] == "european" else np.nan
    price_diff_fd = fd["price"] - bsm["price"] if params["exercise_style"] == "european" else np.nan
    metric_grid(
        [
            ("Black-Scholes-Merton (European)", f"{bsm['price']:.5f}", None),
            ("Binomial price", f"{binom['price']:.5f}", f"vs BSM {price_diff_binom:+.5f}" if np.isfinite(price_diff_binom) else None),
            ("Finite-difference", f"{fd['price']:.5f}", f"vs BSM {price_diff_fd:+.5f}" if np.isfinite(price_diff_fd) else None),
            ("Delta", f"{bsm['delta']:.5f}", None),
            ("Gamma", f"{bsm['gamma']:.5f}", None),
            ("Vega", f"{bsm['vega']:.5f}", None),
            ("Theta / year", f"{bsm['theta']:.5f}", None),
            ("Rho", f"{bsm['rho']:.5f}", None),
        ],
        ncols=4,
    )

    c1, c2 = st.columns([1.2, 1.0])
    with c1:
        price_df = pd.DataFrame(
            {
                "Method": ["Black-Scholes (European)", "Binomial", "Finite difference"],
                "Price": [bsm["price"], binom["price"], fd["price"]],
            }
        )
        st.plotly_chart(make_price_comparison_figure(price_df), use_container_width=True)
    with c2:
        st.plotly_chart(make_lognormal_distribution_figure(params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]), use_container_width=True)

    st.plotly_chart(make_payoff_value_figure(params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], params["option_type"]), use_container_width=True)


def render_model_comparison(params: Dict[str, Any], bsm: Dict[str, float], binom: Dict[str, Any], fd: Dict[str, Any]) -> None:
    section_header("Model Comparison", "Compare the European Black-Scholes and Monte Carlo benchmarks with the selected lattice and PDE exercise style.")
    mc_paths = st.slider("Monte Carlo paths for comparison", 2_000, 120_000, 25_000, 1_000)
    seed = st.number_input("Monte Carlo seed", min_value=1, max_value=999_999, value=12345, step=1)
    mc = monte_carlo_european(params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], params["option_type"], mc_paths, int(seed))
    price_df = pd.DataFrame(
        [
            {"Method": "Black-Scholes-Merton (European)", "Price": bsm["price"], "Error vs BSM": 0.0, "Notes": "Closed-form European benchmark"},
            {"Method": "CRR Binomial", "Price": binom["price"], "Error vs BSM": binom["price"] - bsm["price"] if params["exercise_style"] == "european" else np.nan, "Notes": "Backward induction; early exercise when American"},
            {"Method": fd["method"], "Price": fd["price"], "Error vs BSM": fd["price"] - bsm["price"] if params["exercise_style"] == "european" else np.nan, "Notes": "PDE grid solver"},
            {"Method": "Monte Carlo (European)", "Price": mc["price"], "Error vs BSM": mc["price"] - bsm["price"], "Notes": f"European estimate; 95% CI [{mc['ci_low']:.4f}, {mc['ci_high']:.4f}]"},
        ]
    )
    df_display(price_df)
    if params["exercise_style"] == "american":
        st.info("The Monte Carlo row prices a European payoff at maturity. It does not model early exercise, so compare it with the European Black-Scholes benchmark rather than the American lattice and PDE prices.")
    st.plotly_chart(make_price_comparison_figure(price_df), use_container_width=True)
    st.plotly_chart(make_convergence_figure(params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], params["option_type"], params["exercise_style"], params["binomial_model"]), use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=mc["running"], x=np.arange(1, len(mc["running"]) + 1), mode="lines", name="MC running estimate"))
    fig.add_hline(y=bsm["price"], line_dash="dash", annotation_text=f"BSM={bsm['price']:.4f}")
    fig.update_xaxes(title="Number of paths")
    fig.update_yaxes(title="Estimated price")
    st.plotly_chart(plot_layout(fig, "Monte Carlo convergence to BSM benchmark", height=500), use_container_width=True)


def render_distribution_lab(params: Dict[str, Any], binom: Dict[str, Any], fd: Dict[str, Any]) -> None:
    section_header("Distribution Laboratory", "Continuous and discrete risk-neutral probability distributions with numerical probability tables.")

    tabs = st.tabs(["Terminal density", "Binomial terminal law", "Probability field", "Numerical tables"])

    with tabs[0]:
        st.plotly_chart(
            make_lognormal_distribution_figure(params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]),
            use_container_width=True,
        )
        S_grid = np.linspace(max(0.01, 0.02 * params["S0"]), 2.8 * params["S0"], 220)
        dens = risk_neutral_density(S_grid, params["S0"], params["T"], params["r"], params["q"], params["sigma"])
        dS = float(S_grid[1] - S_grid[0]) if len(S_grid) > 1 else 1.0
        prob_mass = dens * dS
        cumulative = np.concatenate(([0.0], np.cumsum((dens[:-1] + dens[1:]) * 0.5 * dS)))
        terminal_density_table = pd.DataFrame(
            {
                "risk_neutral_density": dens,
                "probability_mass_on_grid": prob_mass,
                "cumulative_probability_on_displayed_grid": cumulative,
            },
            index=[f"S_T={x:.2f}" for x in S_grid],
        )
        show_colored_numeric_table(
            terminal_density_table.iloc[::2],
            precision=8,
            height=560,
            cmap="viridis",
            caption="Untruncated lognormal risk-neutral density. Mass and cumulative probability show only the portion captured by the displayed price grid; omitted tails are not renormalized.",
        )

    with tabs[1]:
   
        terminal_prices = np.asarray(binom["stock_tree"][-1], dtype=float)
        terminal_probs = np.asarray(binom["probability_tree"][-1], dtype=float)
        terminal_values = np.asarray(binom["option_tree"][-1], dtype=float)
        intrinsic = payoff(terminal_prices, params["K"], params["option_type"])

        term_df = pd.DataFrame(
            {
                "terminal_stock_price": terminal_prices,
                "risk_neutral_probability": terminal_probs,
                "terminal_payoff": intrinsic,
                "terminal_option_value": terminal_values,
                "discounted_probability_weight": np.exp(-params["r"] * params["T"]) * terminal_probs,
            }
        ).sort_values("terminal_stock_price")

        plot_df = term_df[term_df["risk_neutral_probability"] > 1e-5].copy()

        if plot_df.empty:
            plot_df = term_df.nlargest(80, "risk_neutral_probability").sort_values("terminal_stock_price")

        bar_width = (
            (plot_df["terminal_stock_price"].max() - plot_df["terminal_stock_price"].min())
            / max(len(plot_df), 1)
            * 0.90
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=plot_df["terminal_stock_price"],
                y=plot_df["risk_neutral_probability"],
                width=bar_width,
                marker=dict(
                    color=plot_df["risk_neutral_probability"],
                    colorscale=COLORSCALE_PROB,
                    showscale=True,
                    colorbar=dict(title="probability"),
                    line=dict(width=0.35, color="rgba(255,255,255,0.25)"),
                ),
                hovertemplate=(
                    "S_T=%{x:.6f}<br>"
                    "probability=%{y:.8f}<extra></extra>"
                ),
                name="Binomial probability",
            )
        )

        fig.add_vline(x=params["K"], line_dash="dash", annotation_text="Strike")

        fig.update_xaxes(
            title="Terminal stock price",
            range=[
                plot_df["terminal_stock_price"].min() * 0.97,
                plot_df["terminal_stock_price"].max() * 1.03,
            ],
        )

        fig.update_yaxes(title="Risk-neutral node probability")

        st.plotly_chart(
            plot_layout(fig, "Discrete terminal distribution of the binomial tree", height=540),
            use_container_width=True,
        )

        show_colored_numeric_table(
            term_df.set_index(
                term_df["terminal_stock_price"].map(lambda x: f"S_T={x:.2f}")
            ).drop(columns=["terminal_stock_price"]),
            precision=8,
            height=560,
            cmap="viridis",
            caption=(
                "Discrete binomial terminal distribution. "
                "Colors encode probabilities, but the numerical values are kept visible."
            ),
        )

    with tabs[2]:
        st.plotly_chart(make_probability_field_figure(fd, params["S0"], params["r"], params["q"], params["sigma"]), use_container_width=True)
        prob_table = probability_numeric_table(fd, params["S0"], params["r"], params["q"], params["sigma"], max_rows=36, max_cols=26)
        show_colored_numeric_table(
            prob_table,
            precision=8,
            height=620,
            cmap="viridis",
            caption="Probability density transported over the finite-difference grid. Rows are S-levels and columns are time-to-maturity levels.",
        )

    with tabs[3]:
        st.markdown("### Probability and value tables")
        st.markdown("These are the numerical tables behind the charts. The color is only an overlay to make the probability concentration immediately visible.")
        table_kind = st.selectbox(
            "Table",
            ["Finite-difference value grid", "Risk-neutral probability field", "Binomial terminal probability", "Terminal continuous density"],
            index=0,
            key="distribution_table_kind",
        )
        if table_kind == "Finite-difference value grid":
            table = finite_difference_numeric_table(fd, "grid", max_rows=38, max_cols=26)
            show_colored_numeric_table(table, precision=5, height=660, cmap="turbo")
        elif table_kind == "Risk-neutral probability field":
            table = probability_numeric_table(fd, params["S0"], params["r"], params["q"], params["sigma"], max_rows=38, max_cols=26)
            show_colored_numeric_table(table, precision=8, height=660, cmap="viridis")
        elif table_kind == "Binomial terminal probability":
            terminal_prices = np.asarray(binom["stock_tree"][-1], dtype=float)
            terminal_probs = np.asarray(binom["probability_tree"][-1], dtype=float)
            table = pd.DataFrame({"risk_neutral_probability": terminal_probs}, index=[f"S_T={x:.2f}" for x in terminal_prices])
            table = table.sort_index()
            show_colored_numeric_table(table, precision=8, height=660, cmap="viridis")
        else:
            S_grid = np.linspace(max(0.01, 0.02 * params["S0"]), 2.8 * params["S0"], 180)
            dens = risk_neutral_density(S_grid, params["S0"], params["T"], params["r"], params["q"], params["sigma"])
            dS = float(S_grid[1] - S_grid[0]) if len(S_grid) > 1 else 1.0
            table = pd.DataFrame(
                {"risk_neutral_density": dens, "probability_mass_on_grid": dens * dS},
                index=[f"S_T={x:.2f}" for x in S_grid],
            )
            show_colored_numeric_table(table.iloc[::2], precision=8, height=660, cmap="viridis")


def render_binomial_lab(params: Dict[str, Any], binom: Dict[str, Any]) -> None:
    section_header("Binomial Tree Laboratory", "Clickable CRR/Jarrow-Rudd lattice with node-level replication parameters.")
    N_display = st.slider("Displayed lattice depth", 3, min(60, params["N_binomial"]), min(22, params["N_binomial"]), 1)
    fig = make_binomial_tree_figure(binom, params["K"], N_display)
    event = plotly_chart_selectable(fig, key="binomial_tree_selectable")
    selected = parse_selected_customdata(event)

    st.markdown("### Node Inspector")
    colA, colB, colC = st.columns([0.8, 0.8, 1.4])
    with colA:
        manual_i = st.number_input("Manual step i", min_value=0, max_value=N_display, value=0, step=1)
    with colB:
        manual_j = st.number_input("Manual node j", min_value=0, max_value=int(manual_i), value=0, step=1)
    with colC:
        source = st.radio("Node source", ["Manual selector", "Last clicked node"], horizontal=True, disabled=selected is None)

    if selected is not None and source == "Last clicked node":
        i, j = int(selected[0]), int(selected[1])
    else:
        i, j = int(manual_i), int(manual_j)

    S = float(binom["stock_tree"][i][j])
    V = float(binom["option_tree"][i][j])
    prob = float(binom["probability_tree"][i][j])
    cont = float(binom["continuation_tree"][i][j]) if len(binom["continuation_tree"][i]) > j and np.isfinite(binom["continuation_tree"][i][j]) else np.nan
    intrinsic = float(binom["intrinsic_tree"][i][j])
    delta = float(binom["delta_tree"][i][j]) if len(binom["delta_tree"][i]) > j and np.isfinite(binom["delta_tree"][i][j]) else np.nan
    bond = float(binom["bond_tree"][i][j]) if len(binom["bond_tree"][i]) > j and np.isfinite(binom["bond_tree"][i][j]) else np.nan
    early = bool(binom["early_exercise_tree"][i][j]) if len(binom["early_exercise_tree"][i]) > j else False

    metric_grid(
        [
            ("Step i", f"{i}", None),
            ("Node j", f"{j}", None),
            ("Stock S(i,j)", f"{S:.6f}", None),
            ("Option V(i,j)", f"{V:.6f}", None),
            ("Node probability", f"{prob:.4%}", None),
            ("Intrinsic value", f"{intrinsic:.6f}", None),
            ("Continuation value", f"{cont:.6f}" if np.isfinite(cont) else "terminal", None),
            ("Early exercise", "YES" if early else "NO", None),
            ("Delta hedge", f"{delta:.6f}" if np.isfinite(delta) else "terminal", None),
            ("Bond position", f"{bond:.6f}" if np.isfinite(bond) else "terminal", None),
        ],
        ncols=5,
    )

    with st.expander("Hull-style one-step replication formula"):
        st.latex(r"\Delta=\frac{f_u-f_d}{S_u-S_d}")
        st.latex(r"B=e^{-r\Delta t}\left(f_u-\Delta S_u\right)")
        st.latex(r"f=\Delta S+B")
        st.markdown("The selected node contains the local hedge ratio and the bond component that replicate the option over the next time step.")


def render_fd_lab(params: Dict[str, Any], fd: Dict[str, Any], bsm: Dict[str, float]) -> None:
    section_header("Finite Difference Laboratory", "PDE grid, Crank-Nicolson solver, American obstacle and probability transport.")
    tabs = st.tabs(["PDE", "Value grid", "Probability field", "3D surface", "Free boundary", "Coefficients", "Numerical table"])

    with tabs[0]:
        st.markdown("### Black-Scholes-Merton PDE")
        st.latex(r"\frac{\partial V}{\partial t}+(r-q)S\frac{\partial V}{\partial S}+\frac12\sigma^2S^2\frac{\partial^2 V}{\partial S^2}-rV=0")
        st.markdown(r"Using time-to-maturity \(\tau=T-t\):")
        st.latex(r"\frac{\partial V}{\partial \tau}=\frac12\sigma^2S^2V_{SS}+(r-q)SV_S-rV")
        st.markdown("For Crank-Nicolson:")
        st.latex(r"(I-\tfrac12\Delta\tau L)V^{n+1}=(I+\tfrac12\Delta\tau L)V^n")
        if params["exercise_style"] == "american":
            st.markdown("American exercise is treated as a linear complementarity problem:")
            st.latex(r"V\geq \Phi,\qquad AV-b\geq 0,\qquad (V-\Phi)^T(AV-b)=0")
        metric_grid(
            [
                ("FD price", f"{fd['price']:.6f}", f"vs BSM {fd['price']-bsm['price']:+.6f}" if params["exercise_style"] == "european" else None),
                ("Grid M", f"{params['M_fd']}", None),
                ("Time N", f"{params['N_fd']}", None),
                ("dS", f"{fd['dS']:.6f}", None),
                ("dtau", f"{fd['dt']:.6f}", None),
                ("PSOR mean iter", f"{fd['psor_mean_iters']:.1f}" if np.isfinite(fd["psor_mean_iters"]) else "NA", None),
            ],
            ncols=3,
        )

    with tabs[1]:
        qty = st.selectbox("Grid quantity", ["grid", "delta_grid", "gamma_grid", "theta_grid", "intrinsic_grid"], index=0)
        st.plotly_chart(make_fd_heatmap(fd, qty, f"Finite-difference {qty.replace('_', ' ')}"), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            step = st.slider("Inspect time index n", 0, len(fd["tau"]) - 1, len(fd["tau"]) - 1)
        with col2:
            sidx = st.slider("Inspect space index i", 0, len(fd["S"]) - 1, int(np.argmin(np.abs(fd["S"] - params["S0"]))))
        metric_grid(
            [
                ("tau", f"{fd['tau'][step]:.6f}", None),
                ("S_i", f"{fd['S'][sidx]:.6f}", None),
                ("V", f"{fd['grid'][step, sidx]:.6f}", None),
                ("Delta", f"{fd['delta_grid'][step, sidx]:.6f}" if np.isfinite(fd['delta_grid'][step, sidx]) else "NA", None),
                ("Gamma", f"{fd['gamma_grid'][step, sidx]:.6f}" if np.isfinite(fd['gamma_grid'][step, sidx]) else "NA", None),
                ("Theta", f"{fd['theta_grid'][step, sidx]:.6f}" if np.isfinite(fd['theta_grid'][step, sidx]) else "NA", None),
            ],
            ncols=3,
        )
        with st.expander("Numerical table for selected grid quantity", expanded=True):
            table = finite_difference_numeric_table(fd, qty, max_rows=34, max_cols=24)
            show_colored_numeric_table(
                table,
                precision=5,
                height=560,
                cmap="turbo",
                caption="Every cell contains the numerical value; the color is only a heatmap overlay.",
            )

    with tabs[2]:
        st.plotly_chart(make_probability_field_figure(fd, params["S0"], params["r"], params["q"], params["sigma"]), use_container_width=True)
        st.markdown("The colors describe where the risk-neutral probability mass is concentrated while the PDE grid evolves backwards from maturity to today.")
        prob_table = probability_numeric_table(fd, params["S0"], params["r"], params["q"], params["sigma"], max_rows=34, max_cols=24)
        show_colored_numeric_table(
            prob_table,
            precision=7,
            height=560,
            cmap="viridis",
            caption="Risk-neutral probability density table. The numbers remain visible in every cell.",
        )

    with tabs[3]:
        st.plotly_chart(make_fd_surface_figure(fd), use_container_width=True)
        st.plotly_chart(make_animated_surface_cross_section(fd), use_container_width=True)

    with tabs[4]:
        if params["exercise_style"] != "american":
            st.info("Switch Exercise style to American to see the free-boundary approximation.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fd["tau"], y=fd["free_boundary"], mode="lines+markers", name="free boundary"))
            fig.update_xaxes(title="tau")
            fig.update_yaxes(title="Critical underlying level S*(tau)")
            st.plotly_chart(plot_layout(fig, "Approximate optimal exercise boundary", height=520), use_container_width=True)
            st.plotly_chart(make_fd_heatmap({**fd, "exercise_region": fd["exercise_region"].astype(float)}, "exercise_region", "American exercise region"), use_container_width=True)

    with tabs[5]:
        i = np.arange(1, params["M_fd"], dtype=int)
        coeff_df = pd.DataFrame(
            {
                "i": i,
                "S_i": fd["S"][1:-1],
                "A0 lower coefficient": fd["A0"],
                "B0 center coefficient": fd["B0"],
                "C0 upper coefficient": fd["C0"],
            }
        )
        df_display(coeff_df, height=420)
        st.markdown(r"These coefficients define the spatial differential operator \(L\) used by implicit and Crank-Nicolson schemes.")

    with tabs[6]:
        st.markdown("### Hull-style finite-difference numerical table")
        table_qty = st.selectbox(
            "Table quantity",
            ["grid", "intrinsic_grid", "delta_grid", "gamma_grid", "theta_grid"],
            index=0,
            key="fd_numerical_table_quantity",
        )
        table = finite_difference_numeric_table(fd, table_qty, max_rows=38, max_cols=26)
        show_colored_numeric_table(
            table,
            precision=5,
            height=660,
            cmap="turbo",
            caption="Rows are underlying levels S_i, columns are time-to-maturity levels tau_n. This is the colored numerical version of the finite-difference table.",
        )


def render_stability_lab(params: Dict[str, Any]) -> None:
    section_header("Stability & Instability Laboratory", "Visual demonstration of how an explicit finite-difference scheme can explode.")
    mode = st.radio("Experiment mode", ["Stable demonstration", "Unstable demonstration", "Custom"], horizontal=True)
    if mode == "Stable demonstration":
        M_stab, N_stab, sig_stab = 45, 450, params["sigma"]
    elif mode == "Unstable demonstration":
        M_stab, N_stab, sig_stab = 95, 25, max(params["sigma"], 0.32)
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            M_stab = st.slider("Spatial points M", 20, 160, 80, 1)
        with c2:
            N_stab = st.slider("Time steps N", 5, 600, 80, 1)
        with c3:
            sig_stab = st.slider("Volatility sigma for stability demo", 0.05, 1.20, params["sigma"], 0.01)

    diag = explicit_stability_coefficients(M_stab, N_stab, params["T"], params["r"], params["q"], sig_stab)
    status = "Stable" if diag["stable"] else "Unstable / high-risk"
    status_class = "status-green" if diag["stable"] else "status-red"
    st.markdown(f"<div class='glass-card'>Explicit scheme status: <span class='{status_class}'>{status}</span></div>", unsafe_allow_html=True)
    metric_grid(
        [
            ("M", f"{M_stab}", None),
            ("N", f"{N_stab}", None),
            ("dtau", f"{diag['dt']:.6f}", None),
            ("min(alpha,beta,gamma)", f"{diag['min_coeff']:.6f}", None),
            ("max CFL proxy", f"{diag['max_cfl']:.4f}", None),
        ],
        ncols=5,
    )

    fd_explicit = finite_difference_solver(
        params["S0"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        sig_stab,
        params["Smax"],
        M_stab,
        N_stab,
        params["option_type"],
        "european",
        "Explicit",
    )
    tabs = st.tabs(["Instability heatmap", "Error growth", "Animated 3D", "Stability map", "Coefficients"])
    with tabs[0]:
        st.plotly_chart(make_fd_heatmap(fd_explicit, "grid", "Explicit finite-difference solution"), use_container_width=True)
        st.markdown("Checkerboard patterns, alternating signs and extreme colors are numerical artifacts, not financial effects.")
        show_colored_numeric_table(
            finite_difference_numeric_table(fd_explicit, "grid", max_rows=34, max_cols=22),
            precision=5,
            height=540,
            cmap="turbo",
            caption="Numerical values of the explicit scheme. In the unstable case, large positive/negative numbers reveal the explosion.",
        )
    with tabs[1]:
        err_fig, err = make_error_heatmap(fd_explicit, params["K"], params["r"], params["q"], sig_stab, params["option_type"])
        st.plotly_chart(err_fig, use_container_width=True)
        st.plotly_chart(make_error_growth_figure(fd_explicit, err), use_container_width=True)
        err_table = make_numeric_grid_table(err.T, fd_explicit["S"], fd_explicit["tau"], row_label="S", col_label="tau", max_rows=34, max_cols=22)
        show_colored_numeric_table(
            err_table,
            precision=5,
            height=540,
            cmap="inferno",
            caption="Absolute error table against Black-Scholes. Colors highlight the explosion, numbers quantify it.",
        )
    with tabs[2]:
        st.plotly_chart(make_animated_surface_cross_section(fd_explicit, clip_quantile=0.98), use_container_width=True)
    with tabs[3]:
        st.plotly_chart(make_stability_map_figure(params["T"], params["r"], params["q"], sig_stab), use_container_width=True)
        st.latex(r"\alpha_i=\tfrac12\Delta\tau(\sigma^2i^2-(r-q)i),\quad \beta_i=1-\Delta\tau(\sigma^2i^2+r),\quad \gamma_i=\tfrac12\Delta\tau(\sigma^2i^2+(r-q)i)")
    with tabs[4]:
        coeff_df = pd.DataFrame({"i": np.arange(1, M_stab), "alpha": diag["alpha"], "beta": diag["beta"], "gamma": diag["gamma"]})
        df_display(coeff_df, height=520)


def render_monte_carlo_lab(params: Dict[str, Any], bsm: Dict[str, float]) -> None:
    section_header("Monte Carlo Laboratory", "Risk-neutral path simulation for a European payoff, terminal distribution and estimator convergence.")
    if params["exercise_style"] == "american":
        st.info("This Monte Carlo engine prices European exercise only. The displayed Black-Scholes benchmark is European too; American early exercise is not simulated.")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_paths = st.slider("Pricing paths", 2_000, 200_000, 50_000, 1_000)
    with c2:
        path_vis = st.slider("Displayed paths", 5, 80, 30, 1)
    with c3:
        seed = st.number_input("Seed", 1, 999_999, 2026, 1)

    mc = monte_carlo_european(params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], params["option_type"], n_paths, int(seed))
    metric_grid(
        [
            ("European MC price", f"{mc['price']:.6f}", f"vs European BSM {mc['price']-bsm['price']:+.6f}"),
            ("Std. error", f"{mc['std_error']:.6f}", None),
            ("95% CI low", f"{mc['ci_low']:.6f}", None),
            ("95% CI high", f"{mc['ci_high']:.6f}", None),
        ],
        ncols=4,
    )
    t, paths = simulate_gbm_paths(params["S0"], params["T"], params["r"], params["q"], params["sigma"], path_vis, 160, int(seed))
    fig_paths = go.Figure()
    for i in range(path_vis):
        fig_paths.add_trace(go.Scatter(x=t, y=paths[i, :], mode="lines", opacity=0.55, showlegend=False, line=dict(width=1.6)))
    fig_paths.add_hline(y=params["K"], line_dash="dash", annotation_text="Strike")
    fig_paths.update_xaxes(title="time")
    fig_paths.update_yaxes(title="S(t)")
    st.plotly_chart(plot_layout(fig_paths, "Risk-neutral geometric Brownian motion paths", height=540), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=mc["ST"], nbinsx=70, histnorm="probability density", name="Simulated ST", marker=dict(line=dict(width=0.5, color="white"))))
        S_grid = np.linspace(max(0.01, np.percentile(mc["ST"], 0.5)), np.percentile(mc["ST"], 99.5), 350)
        fig_hist.add_trace(go.Scatter(x=S_grid, y=risk_neutral_density(S_grid, params["S0"], params["T"], params["r"], params["q"], params["sigma"]), mode="lines", name="Lognormal density", line=dict(width=4)))
        st.plotly_chart(plot_layout(fig_hist, "Terminal distribution: simulation versus lognormal density", height=520), use_container_width=True)
    with c2:
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=np.arange(1, len(mc["running"]) + 1), y=mc["running"], mode="lines", name="Running estimator"))
        fig_conv.add_hline(y=bsm["price"], line_dash="dash", annotation_text=f"BSM={bsm['price']:.4f}")
        fig_conv.update_xaxes(title="paths")
        fig_conv.update_yaxes(title="estimated price")
        st.plotly_chart(plot_layout(fig_conv, "Monte Carlo estimator convergence", height=520), use_container_width=True)


def render_greeks_lab(params: Dict[str, Any], bsm: Dict[str, float], fd: Dict[str, Any]) -> None:
    section_header("Greeks & Sensitivities", "Analytic Greeks and PDE Greeks computed on the finite-difference grid.")
    if params["exercise_style"] == "american":
        st.info("The analytic Black-Scholes Greeks are for a European option. PDE Greeks below use the selected American exercise condition.")
    greeks_df = pd.DataFrame(
        {
            "Greek": ["Delta", "Gamma", "Vega", "Theta", "Rho"],
            "Value": [bsm["delta"], bsm["gamma"], bsm["vega"], bsm["theta"], bsm["rho"]],
            "Meaning": [
                "Sensitivity to S",
                "Curvature with respect to S",
                "Sensitivity to volatility sigma",
                "Time decay per year",
                "Sensitivity to risk-free rate r",
            ],
        }
    )
    df_display(greeks_df)
    greek_choice = st.selectbox("Greek surface", ["delta", "gamma", "vega", "theta", "rho"], index=0)
    st.plotly_chart(make_greek_surface(greek_choice, params["K"], params["T"], params["r"], params["q"], params["sigma"], params["option_type"], params["S0"]), use_container_width=True)

    tabs = st.tabs(["FD Delta", "FD Gamma", "FD Theta"])
    with tabs[0]:
        st.plotly_chart(make_fd_heatmap(fd, "delta_grid", "PDE Delta across grid"), use_container_width=True)
        show_colored_numeric_table(
            finite_difference_numeric_table(fd, "delta_grid", max_rows=34, max_cols=22),
            precision=6,
            height=540,
            cmap="turbo",
            caption="Numerical Delta grid. Every cell contains the finite-difference Delta estimate.",
        )
    with tabs[1]:
        st.plotly_chart(make_fd_heatmap(fd, "gamma_grid", "PDE Gamma across grid"), use_container_width=True)
        show_colored_numeric_table(
            finite_difference_numeric_table(fd, "gamma_grid", max_rows=34, max_cols=22),
            precision=8,
            height=540,
            cmap="turbo",
            caption="Numerical Gamma grid. Values are shown explicitly, with colors as background intensity only.",
        )
    with tabs[2]:
        st.plotly_chart(make_fd_heatmap(fd, "theta_grid", "PDE Theta across grid"), use_container_width=True)
        show_colored_numeric_table(
            finite_difference_numeric_table(fd, "theta_grid", max_rows=34, max_cols=22),
            precision=6,
            height=540,
            cmap="turbo",
            caption="Numerical Theta grid. Negative values indicate time decay over the PDE grid.",
        )


def render_theory(params: Dict[str, Any]) -> None:
    section_header("Theory & Numerical Notes", "Mathematical structure behind the laboratory.")
    st.markdown(
        """
        ### 1. Risk-neutral valuation
        Under the risk-neutral measure, the discounted derivative price is a martingale. The expected return of the underlying is replaced by the risk-free rate adjusted for continuous dividends.
        """
    )
    st.latex(r"dS=(r-q)Sdt+\sigma S dW_t")
    st.latex(r"V_0=e^{-rT}\mathbb{E}^{\mathbb{Q}}[\Phi(S_T)]")

    st.markdown("### 2. Binomial model")
    st.latex(r"u=e^{\sigma\sqrt{\Delta t}},\qquad d=\frac1u,\qquad p=\frac{e^{(r-q)\Delta t}-d}{u-d}")
    st.latex(r"V_{i,j}=e^{-r\Delta t}\left(pV_{i+1,j+1}+(1-p)V_{i+1,j}\right)")

    st.markdown("### 3. Finite differences")
    st.latex(r"\frac{\partial V}{\partial \tau}=\frac12\sigma^2S^2V_{SS}+(r-q)SV_S-rV")
    st.latex(r"(I-\theta\Delta\tau L)V^{n+1}=(I+(1-\theta)\Delta\tau L)V^n")
    st.markdown(r"The parameter \(\theta=0\) gives explicit Euler, \(\theta=1\) gives implicit Euler, and \(\theta=1/2\) gives Crank-Nicolson.")

    st.markdown("### 4. Stability")
    st.markdown("The explicit scheme can become unstable when the time step is too large relative to the spatial step. In coefficient form, the scheme is safe only when the pseudo-probability weights remain non-negative.")
    st.latex(r"V_i^{n+1}=\alpha_iV_{i-1}^{n}+\beta_iV_i^n+\gamma_iV_{i+1}^{n}")
    st.latex(r"\alpha_i,\beta_i,\gamma_i\geq0")

    st.markdown("### 5. American option as an obstacle problem")
    st.latex(r"V(S,t)\geq \Phi(S)")
    st.latex(r"\max\left(\mathcal{L}V,\Phi(S)-V\right)=0")


def render_report_center(params: Dict[str, Any], bsm: Dict[str, float], binom: Dict[str, Any], fd: Dict[str, Any]) -> None:
    section_header("Report Center", "Export a compact numerical report with parameters, prices, Greeks and formulas.")
    prices = {
        "Black-Scholes-Merton (European)": bsm["price"],
        "Binomial": binom["price"],
        "Finite Difference": fd["price"],
    }
    report_params = {
        "S0": params["S0"],
        "K": params["K"],
        "T": params["T"],
        "r": params["r"],
        "q": params["q"],
        "sigma": params["sigma"],
        "option_type": params["option_type"],
        "exercise_style": params["exercise_style"],
        "binomial_steps": params["N_binomial"],
        "FD_method": params["fd_method"],
        "FD_grid": f"M={params['M_fd']}, N={params['N_fd']}",
    }
    md = make_markdown_report(report_params, prices, bsm)
    pdf = minimal_pdf_bytes("Option Pricing Laboratory Report", md.splitlines())
    st.download_button("Download Markdown report", md.encode("utf-8"), file_name="option_pricing_report.md", mime="text/markdown")
    st.download_button("Download PDF report", pdf, file_name="option_pricing_report.pdf", mime="application/pdf")
    st.code(md, language="markdown")
