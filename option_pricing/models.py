"""Pricing formulas and numerical solvers."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

EPS = 1e-12


def normal_pdf(x: float | np.ndarray) -> float | np.ndarray:
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_cdf_vec(x: np.ndarray) -> np.ndarray:
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def payoff(S: np.ndarray | float, K: float, option_type: str) -> np.ndarray | float:
    arr = np.asarray(S)
    if option_type == "call":
        out = np.maximum(arr - K, 0.0)
    else:
        out = np.maximum(K - arr, 0.0)
    if np.isscalar(S):
        return float(out)
    return out


def safe_div(a: float, b: float, fallback: float = 0.0) -> float:
    return a / b if abs(b) > EPS else fallback


def black_scholes_merton(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str = "call",
) -> Dict[str, float]:
    """Black-Scholes-Merton price and Greeks.

    Theta is expressed per year. Vega and rho are raw derivatives with respect
    to sigma and r, not divided by 100.
    """
    S = max(float(S), EPS)
    K = max(float(K), EPS)
    T = max(float(T), EPS)
    sigma = max(float(sigma), EPS)

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if option_type == "call":
        price = S * disc_q * normal_cdf(d1) - K * disc_r * normal_cdf(d2)
        delta = disc_q * normal_cdf(d1)
        theta = (
            -S * disc_q * float(normal_pdf(d1)) * sigma / (2.0 * sqrtT)
            - r * K * disc_r * normal_cdf(d2)
            + q * S * disc_q * normal_cdf(d1)
        )
        rho = K * T * disc_r * normal_cdf(d2)
    else:
        price = K * disc_r * normal_cdf(-d2) - S * disc_q * normal_cdf(-d1)
        delta = disc_q * (normal_cdf(d1) - 1.0)
        theta = (
            -S * disc_q * float(normal_pdf(d1)) * sigma / (2.0 * sqrtT)
            + r * K * disc_r * normal_cdf(-d2)
            - q * S * disc_q * normal_cdf(-d1)
        )
        rho = -K * T * disc_r * normal_cdf(-d2)

    gamma = disc_q * float(normal_pdf(d1)) / (S * sigma * sqrtT)
    vega = S * disc_q * float(normal_pdf(d1)) * sqrtT

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
        "d1": float(d1),
        "d2": float(d2),
    }


def black_scholes_price_vectorized(
    S: np.ndarray,
    K: float,
    tau: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    tau = max(float(tau), EPS)
    sigma = max(float(sigma), EPS)
    safe_S = np.maximum(S, EPS)
    sqrtT = math.sqrt(tau)
    d1 = (np.log(safe_S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if option_type == "call":
        return safe_S * np.exp(-q * tau) * normal_cdf_vec(d1) - K * np.exp(-r * tau) * normal_cdf_vec(d2)
    return K * np.exp(-r * tau) * normal_cdf_vec(-d2) - safe_S * np.exp(-q * tau) * normal_cdf_vec(-d1)


@st.cache_data(show_spinner=False)
def build_binomial_model(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    N: int,
    option_type: str,
    exercise_style: str,
    model: str = "CRR",
) -> Dict[str, Any]:
    N = max(int(N), 1)
    dt = T / N

    if model == "CRR":
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
    elif model == "Jarrow-Rudd":
        drift = (r - q - 0.5 * sigma ** 2) * dt
        u = math.exp(drift + sigma * math.sqrt(dt))
        d = math.exp(drift - sigma * math.sqrt(dt))
    else:
        raise ValueError("Unknown binomial model.")

    growth = math.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 <= p <= 1.0):
        raise ValueError(
            "Risk-neutral probability outside [0,1]. Increase N or adjust r, q and sigma."
        )

    discount = math.exp(-r * dt)
    stock_tree: List[np.ndarray] = []
    probability_tree: List[np.ndarray] = []

    for i in range(N + 1):
        stocks = np.array([S0 * (u ** j) * (d ** (i - j)) for j in range(i + 1)], dtype=float)
        probs = np.array(
            [math.comb(i, j) * (p ** j) * ((1.0 - p) ** (i - j)) for j in range(i + 1)],
            dtype=float,
        )
        stock_tree.append(stocks)
        probability_tree.append(probs)

    option_tree: List[np.ndarray] = [np.array([])] * (N + 1)
    continuation_tree: List[np.ndarray] = [np.array([])] * (N + 1)
    intrinsic_tree: List[np.ndarray] = [np.array([])] * (N + 1)
    delta_tree: List[np.ndarray] = [np.array([])] * (N + 1)
    bond_tree: List[np.ndarray] = [np.array([])] * (N + 1)
    early_tree: List[np.ndarray] = [np.array([], dtype=bool)] * (N + 1)

    option_tree[N] = payoff(stock_tree[N], K, option_type)
    intrinsic_tree[N] = payoff(stock_tree[N], K, option_type)
    continuation_tree[N] = np.full(N + 1, np.nan)
    delta_tree[N] = np.full(N + 1, np.nan)
    bond_tree[N] = np.full(N + 1, np.nan)
    early_tree[N] = np.zeros(N + 1, dtype=bool)

    for i in range(N - 1, -1, -1):
        cont = discount * (p * option_tree[i + 1][1:] + (1.0 - p) * option_tree[i + 1][:-1])
        intrinsic = payoff(stock_tree[i], K, option_type)
        if exercise_style == "american":
            values = np.maximum(cont, intrinsic)
            early = intrinsic > cont + 1e-12
        else:
            values = cont
            early = np.zeros(i + 1, dtype=bool)

        deltas = np.empty(i + 1, dtype=float)
        bonds = np.empty(i + 1, dtype=float)
        for j in range(i + 1):
            Su = stock_tree[i + 1][j + 1]
            Sd = stock_tree[i + 1][j]
            Vu = option_tree[i + 1][j + 1]
            Vd = option_tree[i + 1][j]
            delta = safe_div(Vu - Vd, Su - Sd, np.nan)
            bond = math.exp(-r * dt) * (Vu - delta * Su)
            deltas[j] = delta
            bonds[j] = bond

        option_tree[i] = values
        continuation_tree[i] = cont
        intrinsic_tree[i] = intrinsic
        early_tree[i] = early
        delta_tree[i] = deltas
        bond_tree[i] = bonds

    return {
        "price": float(option_tree[0][0]),
        "u": float(u),
        "d": float(d),
        "p": float(p),
        "dt": float(dt),
        "discount": float(discount),
        "stock_tree": stock_tree,
        "probability_tree": probability_tree,
        "option_tree": option_tree,
        "continuation_tree": continuation_tree,
        "intrinsic_tree": intrinsic_tree,
        "early_exercise_tree": early_tree,
        "delta_tree": delta_tree,
        "bond_tree": bond_tree,
    }


def tridiagonal_solve(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal system.

    lower, upper have length n-1; diag and rhs length n.
    """
    n = len(diag)
    if n == 1:
        return rhs / diag
    c = upper.astype(float).copy()
    d = diag.astype(float).copy()
    b = rhs.astype(float).copy()
    a = lower.astype(float).copy()

    for i in range(1, n):
        m = a[i - 1] / d[i - 1]
        d[i] -= m * c[i - 1]
        b[i] -= m * b[i - 1]

    x = np.empty(n, dtype=float)
    x[-1] = b[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (b[i] - c[i] * x[i + 1]) / d[i]
    return x


def psor_lcp(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    obstacle: np.ndarray,
    initial: np.ndarray,
    omega: float = 1.25,
    tol: float = 1e-9,
    max_iter: int = 700,
) -> Tuple[np.ndarray, int, float]:
    """Projected SOR for a tridiagonal linear complementarity problem."""
    x = np.maximum(initial.copy(), obstacle)
    n = len(x)
    last_err = np.inf
    for it in range(1, max_iter + 1):
        err = 0.0
        for i in range(n):
            left = lower[i - 1] * x[i - 1] if i > 0 else 0.0
            right = upper[i] * x[i + 1] if i < n - 1 else 0.0
            y = (rhs[i] - left - right) / diag[i]
            new_x = max(obstacle[i], x[i] + omega * (y - x[i]))
            err = max(err, abs(new_x - x[i]))
            x[i] = new_x
        last_err = err
        if err < tol:
            return x, it, last_err
    return x, max_iter, last_err


def boundary_values(
    K: float,
    tau: float,
    r: float,
    q: float,
    Smax: float,
    option_type: str,
    exercise_style: str,
) -> Tuple[float, float]:
    if option_type == "call":
        lower = 0.0
        european_upper = Smax * math.exp(-q * tau) - K * math.exp(-r * tau)
        upper = max(european_upper, Smax - K if exercise_style == "american" else european_upper, 0.0)
    else:
        european_lower = K * math.exp(-r * tau)
        lower = max(european_lower, K if exercise_style == "american" else european_lower, 0.0)
        upper = 0.0
    return float(lower), float(upper)


@st.cache_data(show_spinner=False)
def finite_difference_solver(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    Smax: float,
    M: int,
    N: int,
    option_type: str,
    exercise_style: str,
    method: str,
    omega: float = 1.25,
) -> Dict[str, Any]:
    """Finite-difference solution in time-to-maturity tau.

    V[0, :] is the payoff at tau=0. The final row V[-1, :] is the present value
    for tau=T. Price is interpolated at S0.
    """
    M = max(int(M), 6)
    N = max(int(N), 2)
    Smax = max(float(Smax), max(S0, K) * 1.25)
    dt = T / N
    dS = Smax / M
    S = np.linspace(0.0, Smax, M + 1)
    tau = np.linspace(0.0, T, N + 1)
    grid = np.zeros((N + 1, M + 1), dtype=float)
    grid[0, :] = payoff(S, K, option_type)

    theta = {"Explicit": 0.0, "Implicit": 1.0, "Crank-Nicolson": 0.5}.get(method, 0.5)
    interior = np.arange(1, M, dtype=float)
    A0 = 0.5 * sigma ** 2 * interior ** 2 - 0.5 * (r - q) * interior
    B0 = -(sigma ** 2 * interior ** 2 + r)
    C0 = 0.5 * sigma ** 2 * interior ** 2 + 0.5 * (r - q) * interior

    psor_iters: List[int] = []
    psor_errors: List[float] = []

    if theta > 0.0:
        lower = -theta * dt * A0[1:]
        diag = 1.0 - theta * dt * B0
        upper = -theta * dt * C0[:-1]
    else:
        lower = diag = upper = np.array([], dtype=float)

    for n in range(N):
        tau_old = tau[n]
        tau_new = tau[n + 1]
        left_old, right_old = boundary_values(K, tau_old, r, q, Smax, option_type, exercise_style)
        left_new, right_new = boundary_values(K, tau_new, r, q, Smax, option_type, exercise_style)
        grid[n, 0] = left_old
        grid[n, M] = right_old

        old = grid[n, :]
        if theta == 0.0:
            new_interior = (
                old[1:M]
                + dt * (A0 * old[0:M - 1] + B0 * old[1:M] + C0 * old[2:M + 1])
            )
            new_row = np.empty(M + 1, dtype=float)
            new_row[0], new_row[M] = left_new, right_new
            new_row[1:M] = new_interior
            if exercise_style == "american":
                new_row = np.maximum(new_row, payoff(S, K, option_type))
            grid[n + 1, :] = new_row
        else:
            rhs = old[1:M] + (1.0 - theta) * dt * (
                A0 * old[0:M - 1] + B0 * old[1:M] + C0 * old[2:M + 1]
            )
            rhs[0] += theta * dt * A0[0] * left_new
            rhs[-1] += theta * dt * C0[-1] * right_new

            if exercise_style == "american":
                obstacle = payoff(S[1:M], K, option_type)
                guess = np.maximum(old[1:M], obstacle)
                solved, iters, err = psor_lcp(lower, diag, upper, rhs, obstacle, guess, omega=omega)
                psor_iters.append(iters)
                psor_errors.append(err)
            else:
                solved = tridiagonal_solve(lower, diag, upper, rhs)

            grid[n + 1, 0] = left_new
            grid[n + 1, M] = right_new
            grid[n + 1, 1:M] = solved

    # Ensure final boundary is consistent
    left_final, right_final = boundary_values(K, T, r, q, Smax, option_type, exercise_style)
    grid[-1, 0] = left_final
    grid[-1, M] = right_final

    price = float(np.interp(S0, S, grid[-1, :]))

    delta_grid = np.full_like(grid, np.nan)
    gamma_grid = np.full_like(grid, np.nan)
    theta_grid = np.full_like(grid, np.nan)
    delta_grid[:, 1:M] = (grid[:, 2:M + 1] - grid[:, 0:M - 1]) / (2.0 * dS)
    gamma_grid[:, 1:M] = (grid[:, 2:M + 1] - 2.0 * grid[:, 1:M] + grid[:, 0:M - 1]) / (dS ** 2)
    if N > 0:
        # tau = T - t, so financial theta dV/dt is -dV/dtau.
        theta_grid[0:N, :] = -(grid[1:N + 1, :] - grid[0:N, :]) / dt

    intrinsic = np.tile(payoff(S, K, option_type), (N + 1, 1))
    exercise_region = (np.abs(grid - intrinsic) < 1e-5) & (intrinsic > 1e-8)

    free_boundary = np.full(N + 1, np.nan)
    if exercise_style == "american":
        for n in range(N + 1):
            idx = np.where(exercise_region[n, :])[0]
            if len(idx) > 0:
                if option_type == "put":
                    free_boundary[n] = S[idx.max()]
                else:
                    free_boundary[n] = S[idx.min()]

    return {
        "price": price,
        "S": S,
        "tau": tau,
        "grid": grid,
        "delta_grid": delta_grid,
        "gamma_grid": gamma_grid,
        "theta_grid": theta_grid,
        "intrinsic_grid": intrinsic,
        "exercise_region": exercise_region,
        "free_boundary": free_boundary,
        "dS": float(dS),
        "dt": float(dt),
        "method": method,
        "psor_mean_iters": float(np.mean(psor_iters)) if psor_iters else np.nan,
        "psor_max_error": float(np.max(psor_errors)) if psor_errors else np.nan,
        "A0": A0,
        "B0": B0,
        "C0": C0,
    }


@st.cache_data(show_spinner=False)
def monte_carlo_european(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
    n_paths: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n_paths = max(int(n_paths), 1000)
    z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)
    pay = payoff(ST, K, option_type)
    discounted = math.exp(-r * T) * pay
    price = float(np.mean(discounted))
    std = float(np.std(discounted, ddof=1))
    se = std / math.sqrt(n_paths)
    ci_low = price - 1.96 * se
    ci_high = price + 1.96 * se
    running = np.cumsum(discounted) / np.arange(1, n_paths + 1)
    return {
        "price": price,
        "std_error": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ST": ST,
        "discounted_payoffs": discounted,
        "running": running,
    }


@st.cache_data(show_spinner=False)
def simulate_gbm_paths(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_paths: int,
    n_steps: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_paths = max(1, int(n_paths))
    n_steps = max(2, int(n_steps))
    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    z = rng.standard_normal((n_paths, n_steps))
    increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(log_paths)
    return t, paths


def risk_neutral_density(S: np.ndarray, S0: float, tau: float, r: float, q: float, sigma: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    out = np.zeros_like(S)
    if tau <= 1e-12:
        idx = int(np.argmin(np.abs(S - S0)))
        if len(S) > 1:
            # Approximate the time-zero Dirac mass with unit area on this grid.
            cell_width = float(np.gradient(S)[idx])
            out[idx] = 1.0 / cell_width if cell_width > 0 else 1.0
        else:
            out[idx] = 1.0
        return out
    positive = S > 0
    x = S[positive]
    mean = math.log(max(S0, EPS)) + (r - q - 0.5 * sigma ** 2) * tau
    var = sigma ** 2 * tau
    out[positive] = (1.0 / (x * math.sqrt(2.0 * math.pi * var))) * np.exp(-((np.log(x) - mean) ** 2) / (2.0 * var))
    return out


def probability_field(S: np.ndarray, tau: np.ndarray, S0: float, r: float, q: float, sigma: float) -> np.ndarray:
    P = np.zeros((len(tau), len(S)), dtype=float)
    for n, t in enumerate(tau):
        P[n, :] = risk_neutral_density(S, S0, float(t), r, q, sigma)
    return P


def explicit_stability_coefficients(M: int, N: int, T: float, r: float, q: float, sigma: float) -> Dict[str, Any]:
    M = max(int(M), 3)
    N = max(int(N), 1)
    dt = T / N
    i = np.arange(1, M, dtype=float)
    alpha = 0.5 * dt * (sigma ** 2 * i ** 2 - (r - q) * i)
    beta = 1.0 - dt * (sigma ** 2 * i ** 2 + r)
    gamma = 0.5 * dt * (sigma ** 2 * i ** 2 + (r - q) * i)
    min_coeff = float(np.min([np.min(alpha), np.min(beta), np.min(gamma)]))
    max_cfl = float(np.max(dt * (sigma ** 2 * i ** 2 + abs(r))))
    stable = bool(min_coeff >= -1e-4 and max_cfl <= 1.0 + 1e-12)
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "min_coeff": min_coeff, "max_cfl": max_cfl, "stable": stable, "dt": dt}


def stability_map_data(
    M_values: np.ndarray,
    N_values: np.ndarray,
    T: float,
    r: float,
    q: float,
    sigma: float,
) -> np.ndarray:
    Z = np.zeros((len(N_values), len(M_values)), dtype=float)
    for iy, N in enumerate(N_values):
        for ix, M in enumerate(M_values):
            diag = explicit_stability_coefficients(int(M), int(N), T, r, q, sigma)
            Z[iy, ix] = diag["min_coeff"]
    return Z
