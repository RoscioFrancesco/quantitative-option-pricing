# Quantitative Option Pricing

Professional Streamlit application for quantitative finance and numerical option pricing.

The project is designed as a university-level workspace for studying option pricing through multiple numerical and analytical engines. Black-Scholes-Merton and Monte Carlo provide European-option benchmarks; the lattice and PDE solvers support both European and American exercise:

- Black-Scholes-Merton closed-form benchmark;
- CRR and Jarrow-Rudd binomial trees;
- node-level binomial inspector with hedge ratio and bond position;
- finite-difference PDE solver;
- explicit, implicit and Crank-Nicolson schemes;
- American option obstacle approximation via PSOR;
- stability and instability diagnostics for explicit finite differences;
- Monte Carlo simulation and convergence diagnostics;
- Greeks and PDE sensitivity grids;
- risk-neutral probability distributions;
- numerical tables with heatmap overlays;
- report export.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS, after the first-time setup, you can also launch with:

```bash
bash scripts/setup_mac.sh
./run.command
```

On Windows, first run `scripts/setup_windows.bat`, then start the app with `scripts/run_windows.bat`. To launch the alternate entry point directly, use `streamlit run quantitative_option_pricing.py`.

## Main files

```text
app.py                          Streamlit entry point
option_pricing/models.py        Pricing formulas and numerical solvers
option_pricing/charts.py        Plotly chart builders
option_pricing/components.py    Shared Streamlit layout and table components
option_pricing/pages.py         Laboratory pages and page rendering
option_pricing/reports.py       Markdown and PDF report generation
quantitative_option_pricing.py  Alternate Streamlit launcher
stability_instability_lab.py    Standalone explicit-scheme instability demo
scripts/                        macOS and Windows setup/launch commands
tests/                          Smoke tests
requirements.txt                Runtime dependencies
requirements-dev.txt            Development dependencies
.streamlit/config.toml          Streamlit theme
```

## Numerical note

All important tabular outputs are numerical tables. Colors are used only as a visual background layer to indicate value intensity, probability mass, error magnitude or Greek sensitivity. They do not replace the numerical values.

## Suggested sections to inspect

1. **Distribution Laboratory** for continuous and binomial risk-neutral probability tables.
2. **Finite Difference Laboratory** for the Hull-style numerical grid.
3. **Stability Laboratory** for explicit-scheme instability and animated 3D visualization.
4. **Binomial Tree Laboratory** for clickable node-level inspection.
