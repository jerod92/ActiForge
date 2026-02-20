"""
Pytest fixtures — shared test infrastructure.

Generates a small in-memory dataset that all tests can use without hitting disk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the package is importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.fixtures.generate_sample_data import generate_all
from auto_actuary.core.session import ActuarySession
from auto_actuary.analytics.triangles.development import LossTriangle


# ---------------------------------------------------------------------------
# Small synthetic triangle (hand-crafted for deterministic tests)
# ---------------------------------------------------------------------------

TRIANGLE_DATA = {
    12:    [1_000_000, 1_100_000, 1_200_000, 1_350_000, 950_000],
    24:    [1_350_000, 1_430_000, 1_510_000, 1_680_000, None],
    36:    [1_450_000, 1_530_000, 1_620_000, None,       None],
    48:    [1_480_000, 1_560_000, None,       None,       None],
    60:    [1_490_000, None,       None,       None,       None],
}
TRIANGLE_ORIGINS = [2019, 2020, 2021, 2022, 2023]


@pytest.fixture(scope="session")
def small_triangle() -> LossTriangle:
    """A small, fully deterministic triangle for unit tests."""
    df = pd.DataFrame(TRIANGLE_DATA, index=TRIANGLE_ORIGINS)
    df.index.name = "accident_year"
    return LossTriangle(df, lob="PPA", value_type="incurred_loss")


@pytest.fixture(scope="session")
def developed_triangle(small_triangle) -> LossTriangle:
    """Triangle with .develop() already called."""
    small_triangle.develop(ldf_method="vw_all", tail_method="user_specified", user_tail=1.010)
    return small_triangle


@pytest.fixture(scope="session")
def sample_data_dir(tmp_path_factory) -> Path:
    """Generate a small CSV dataset in a temp directory."""
    tmpdir = tmp_path_factory.mktemp("sample_data")
    generate_all(output_dir=str(tmpdir), n_per_year=150)
    return tmpdir


@pytest.fixture(scope="session")
def sample_session(sample_data_dir) -> ActuarySession:
    """
    Fully loaded ActuarySession using the generated sample data.
    Uses the default config (which maps canonical names directly — sample data
    uses the same canonical names output by the generator).
    """
    from auto_actuary.core.config import ActuaryConfig

    # Build a minimal in-memory config (no real schema.yaml needed for tests)
    # The generated CSVs use DB column names matching the defaults in schema.yaml
    session = ActuarySession.from_config(
        schema_path=Path(__file__).parent.parent / "config" / "schema.yaml"
    )
    session.load_csv("policies",     sample_data_dir / "policies.csv")
    session.load_csv("claims",       sample_data_dir / "claims.csv")
    session.load_csv("valuations",   sample_data_dir / "valuations.csv")
    session.load_csv("rate_changes", sample_data_dir / "rate_changes.csv")
    session.load_csv("expenses",     sample_data_dir / "expenses.csv")
    return session
