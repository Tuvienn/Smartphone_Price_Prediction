"""
src/inference/predict.py

Inference layer cho Smartphone Price Prediction V1.
Load model artifact một lần, validate input, và trả về giá dự đoán.

Nguyên tắc:
- Model artifact tự chứa toàn bộ preprocessing pipeline (Imputer, Scaler, OneHot, Ridge).
- Không tái hiện bất kỳ bước preprocessing nào bên ngoài artifact.
- Negative prediction được giữ nguyên (raw), không âm thầm sửa.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "models", "smartphone_price_model_v1.joblib"
)

VALID_HAS_5G = {"Yes", "No", "Unknown"}

FEATURE_COLUMNS = [
    "brand",
    "ram_gb",
    "storage_gb",
    "screen_size_inch",
    "has_5g",
    "main_camera_mp",
]

_model = None


def load_model():
    global _model
    if _model is None:
        path = os.path.normpath(MODEL_PATH)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model artifact not found: {path}")
        _model = joblib.load(path)
    return _model


def validate_input(data: dict) -> None:
    errors = []

    # brand
    brand = data.get("brand")
    if not isinstance(brand, str) or not brand.strip():
        errors.append("'brand' must be a non-empty string.")

    # ram_gb — required
    ram_gb = data.get("ram_gb")
    if ram_gb is None:
        errors.append("'ram_gb' is required and must be a positive number.")
    else:
        try:
            if float(ram_gb) <= 0:
                errors.append("'ram_gb' must be > 0.")
        except (TypeError, ValueError):
            errors.append("'ram_gb' must be a numeric value > 0.")

    # storage_gb
    storage_gb = data.get("storage_gb")
    if storage_gb is None:
        errors.append("'storage_gb' is required and must be a positive number.")
    else:
        try:
            if float(storage_gb) <= 0:
                errors.append("'storage_gb' must be > 0.")
        except (TypeError, ValueError):
            errors.append("'storage_gb' must be a numeric value > 0.")

    # screen_size_inch
    screen = data.get("screen_size_inch")
    if screen is None:
        errors.append("'screen_size_inch' is required and must be a positive number.")
    else:
        try:
            if float(screen) <= 0:
                errors.append("'screen_size_inch' must be > 0.")
        except (TypeError, ValueError):
            errors.append("'screen_size_inch' must be a numeric value > 0.")

    # has_5g
    has_5g = data.get("has_5g")
    if has_5g not in VALID_HAS_5G:
        errors.append(f"'has_5g' must be one of {sorted(VALID_HAS_5G)}. Got: {has_5g!r}")

    # main_camera_mp — optional
    camera = data.get("main_camera_mp")
    if camera is not None:
        try:
            if float(camera) <= 0:
                errors.append("'main_camera_mp' must be > 0 if provided.")
        except (TypeError, ValueError):
            errors.append("'main_camera_mp' must be a numeric value > 0, or null/omitted.")

    if errors:
        raise ValueError("Input validation failed:\n  - " + "\n  - ".join(errors))


def predict_price(input_data: dict) -> dict:
    validate_input(input_data)

    model = load_model()

    row = {
        "brand":            str(input_data["brand"]).strip(),
        "ram_gb":           float(input_data["ram_gb"]),
        "storage_gb":       float(input_data["storage_gb"]),
        "screen_size_inch": float(input_data["screen_size_inch"]),
        "has_5g":           input_data["has_5g"],
        "main_camera_mp":   float(input_data["main_camera_mp"]) if input_data.get("main_camera_mp") is not None else None,
    }

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    raw = float(model.predict(df)[0])

    warning = None
    display_price = raw

    if raw < 0:
        display_price = 0.0
        warning = (
            f"Prediction is outside the valid business range "
            f"(raw = {raw:,.0f} VNĐ). "
            f"Model V1 (Ridge, Raw target) does not guarantee non-negative "
            f"output for extreme low-spec inputs."
        )

    return {
        "raw_prediction_vnd":  raw,
        "display_price_vnd":   display_price,
        "formatted_price":     f"{display_price:,.0f} VNĐ",
        "warning":             warning,
    }


if __name__ == "__main__":
    example = {
        "brand":            "Samsung",
        "ram_gb":           12,
        "storage_gb":       256,
        "screen_size_inch": 6.7,
        "has_5g":           "Yes",
        "main_camera_mp":   50,
    }

    print("── Input ──")
    for k, v in example.items():
        print(f"  {k}: {v}")

    result = predict_price(example)

    print("\n── Prediction ──")
    print(f"  Raw prediction : {result['raw_prediction_vnd']:>15,.0f} VNĐ")
    print(f"  Display price  : {result['display_price_vnd']:>15,.0f} VNĐ")
    print(f"  Formatted      : {result['formatted_price']}")
    if result["warning"]:
        print(f"\n  ⚠️  Warning: {result['warning']}")
