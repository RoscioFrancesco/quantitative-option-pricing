import ast
from pathlib import Path


def test_app_syntax_is_valid():
    source = Path(__file__).resolve().parents[1] / "app.py"
    ast.parse(source.read_text())


def test_professional_launcher_exists():
    source = Path(__file__).resolve().parents[1] / "quantitative_option_pricing.py"
    assert source.exists()
