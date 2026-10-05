# Project Manifest

Application entry point: `app.py`
Application modules: `option_pricing/`
Alternate launcher: `quantitative_option_pricing.py`
Optional standalone instability demo: `stability_instability_lab.py`

Module responsibilities:

- `option_pricing/models.py`: analytical formulas, binomial pricing, finite differences, Monte Carlo and stability diagnostics.
- `option_pricing/charts.py`: Plotly figures for prices, distributions, lattices, PDE grids and Greeks.
- `option_pricing/components.py`: shared styling, metrics, and numerical table components.
- `option_pricing/pages.py`: Streamlit renderers for each laboratory section.
- `option_pricing/reports.py`: Markdown and PDF report generation.

Core sections:

- Executive Dashboard
- Model Comparison
- Distribution Laboratory
- Binomial Tree Laboratory
- Finite Difference Laboratory
- Stability Laboratory
- Monte Carlo Laboratory
- Greeks Laboratory
- Theory
- Report Center

Important correction in this release:

- No `height=None` is passed to `st.dataframe`, avoiding `StreamlitInvalidHeightError`.
- Numerical tables contain visible numbers in the cells; colors are only background heatmap overlays.
