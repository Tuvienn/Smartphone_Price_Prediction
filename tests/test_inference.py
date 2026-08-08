import pytest
import sys
import os
from unittest.mock import patch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.inference.predict import load_model, predict_price, validate_input, VALID_HAS_5G

VALID_INPUT = {
    "brand":            "Samsung",
    "ram_gb":           12,
    "storage_gb":       256,
    "screen_size_inch": 6.7,
    "has_5g":           "Yes",
    "main_camera_mp":   50,
}


# ── 1. Load model ──────────────────────────────────────────────────────────
def test_load_model_success():
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict"), "Loaded object must have a predict() method"


# ── 2. Valid input returns dict ────────────────────────────────────────────
def test_valid_input_returns_dict():
    result = predict_price(VALID_INPUT)
    assert isinstance(result, dict)
    for key in ("raw_prediction_vnd", "display_price_vnd", "formatted_price", "warning"):
        assert key in result, f"Missing key in output: {key}"


# ── 3. Prediction is numeric ──────────────────────────────────────────────
def test_prediction_is_numeric():
    result = predict_price(VALID_INPUT)
    assert isinstance(result["raw_prediction_vnd"], float)
    assert isinstance(result["display_price_vnd"], float)


# ── 4. Unknown brand does not crash ──────────────────────────────────────
def test_unknown_brand_no_crash():
    inp = {**VALID_INPUT, "brand": "NewBrandXYZ"}
    result = predict_price(inp)
    assert isinstance(result["raw_prediction_vnd"], float)


# ── 5. Missing camera is handled (optional field) ─────────────────────────
def test_missing_camera_handled():
    inp = {**VALID_INPUT, "main_camera_mp": None}
    result = predict_price(inp)
    assert isinstance(result["raw_prediction_vnd"], float)


# ── 6. has_5g = "Unknown" is handled ─────────────────────────────────────
def test_has_5g_unknown_handled():
    inp = {**VALID_INPUT, "has_5g": "Unknown"}
    result = predict_price(inp)
    assert isinstance(result["raw_prediction_vnd"], float)


# ── 7. Invalid ram_gb raises ValueError ──────────────────────────────────
@pytest.mark.parametrize("bad_ram", [0, -1, -0.5, "abc", None])
def test_invalid_ram_raises(bad_ram):
    inp = {**VALID_INPUT, "ram_gb": bad_ram}
    with pytest.raises(ValueError, match="ram_gb"):
        validate_input(inp)


# ── 8. Invalid storage_gb raises ValueError ──────────────────────────────
@pytest.mark.parametrize("bad_storage", [0, -64, "big"])
def test_invalid_storage_raises(bad_storage):
    inp = {**VALID_INPUT, "storage_gb": bad_storage}
    with pytest.raises(ValueError, match="storage_gb"):
        validate_input(inp)


# ── 9. Invalid has_5g raises ValueError ──────────────────────────────────
@pytest.mark.parametrize("bad_5g", ["maybe", "yes", "NO", 1, True, None])
def test_invalid_has_5g_raises(bad_5g):
    inp = {**VALID_INPUT, "has_5g": bad_5g}
    with pytest.raises(ValueError, match="has_5g"):
        validate_input(inp)


# ── 10. Negative prediction: warning present, display >= 0, raw < 0 ───────
def test_negative_prediction_guardrail():
    with patch("src.inference.predict.load_model") as mock_load:
        mock_model = mock_load.return_value
        mock_model.predict.return_value = np.array([-5_000_000.0])

        result = predict_price(VALID_INPUT)

    assert result["raw_prediction_vnd"] < 0, "raw_prediction_vnd should be negative"
    assert result["display_price_vnd"] == 0.0, "display_price_vnd should be 0 when raw < 0"
    assert result["warning"] is not None, "warning must be set when prediction is negative"
    assert "outside the valid business range" in result["warning"]
