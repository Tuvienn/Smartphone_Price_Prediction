import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.features.pipeline import build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'brand': ['Apple', 'Samsung', 'UnknownBrand', 'Oppo', 'Xiaomi'],
        'ram_gb': [8.0, np.nan, 12.0, 8.0, np.nan],
        'storage_gb': [128.0, 256.0, 512.0, 128.0, 256.0],
        'screen_size_inch': [6.1, 6.8, 6.5, 6.4, 6.7],
        'has_5g': ['Yes', 'No', 'Unknown', 'Yes', 'Unknown'],
        'main_camera_mp': [48.0, np.nan, 50.0, 64.0, 108.0]
    })

def test_pipeline_builds_and_handles_missing(sample_data):
    # Test linear pipeline
    preprocessor = build_preprocessor('linear')
    transformed = preprocessor.fit_transform(sample_data)
    
    # Check shape: 
    # numeric: ram, storage, screen, camera (4) + missing indicators for ram and camera (2) -> 6
    # categorical: brand (5 unique in sample -> 5), has_5g (3 unique -> 3) -> 8
    # Total = 6 + 8 = 14 features
    assert transformed.shape[0] == 5
    assert not np.isnan(transformed).any(), "Imputer failed to fill NaNs"

def test_unknown_brand_does_not_crash(sample_data):
    preprocessor = build_preprocessor('linear')
    preprocessor.fit(sample_data)
    
    new_data = pd.DataFrame({
        'brand': ['CompletelyNewBrand'],
        'ram_gb': [4.0],
        'storage_gb': [64.0],
        'screen_size_inch': [5.0],
        'has_5g': ['Unknown'],
        'main_camera_mp': [12.0]
    })
    
    # Should not raise exception thanks to handle_unknown='ignore'
    transformed = preprocessor.transform(new_data)
    assert transformed.shape[0] == 1

def test_model_predict_numeric(sample_data):
    y = np.array([20000, 30000, 40000, 15000, 25000])
    preprocessor = build_preprocessor('tree')
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    pipe.fit(sample_data, y)
    preds = pipe.predict(sample_data)
    
    assert len(preds) == 5
    assert np.issubdtype(preds.dtype, np.number)

def test_transformed_target_regressor(sample_data):
    y = np.array([20000, 30000, 40000, 15000, 25000])
    preprocessor = build_preprocessor('linear')
    regressor = TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1
    )
    pipe = Pipeline([('preprocessor', preprocessor), ('regressor', regressor)])
    pipe.fit(sample_data, y)
    
    preds = pipe.predict(sample_data)
    # Output should be in original scale, ~ 15000-40000
    assert (preds > 1000).all()

def test_group_kfold_overlap():
    X = pd.DataFrame({'feature': range(10)})
    y = np.array(range(10))
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    
    gkf = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf.split(X, y, groups):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert len(train_groups.intersection(val_groups)) == 0
