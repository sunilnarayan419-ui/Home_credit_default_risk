"""The API schema, the feature-engineering code and the training columns must describe the same inputs."""
import pytest
from conftest import make_frame

from hcdr.api.schemas import RAW_COLUMNS, ApplicationFeatures
from hcdr.config import CAT_COLS

pytestmark = pytest.mark.api


def test_schema_fields_equal_raw_training_columns():
    raw = [c for c in make_frame(5).columns if c not in ("SK_ID_CURR", "TARGET")]
    assert set(RAW_COLUMNS) == set(raw) and len(RAW_COLUMNS) == len(raw) == 32


def test_categorical_columns_are_text_fields():
    for c in CAT_COLS:
        assert c in ApplicationFeatures.model_fields


def test_schema_never_exposes_label_or_id():
    assert "TARGET" not in RAW_COLUMNS and "SK_ID_CURR" not in RAW_COLUMNS


def test_synthetic_rows_validate_against_schema():
    from conftest import records

    for r in records(20, seed=9):
        ApplicationFeatures(**r)
