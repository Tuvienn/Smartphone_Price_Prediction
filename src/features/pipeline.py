import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor(model_type='linear'):
    """
    Builds the preprocessing pipeline according to Phase 5 Plan.
    
    Args:
        model_type (str): 'linear' for linear models (includes StandardScaler) 
                          'tree' for tree-based models (no StandardScaler)
    Returns:
        ColumnTransformer: The configured preprocessor
    """
    numeric_features = ['ram_gb', 'storage_gb', 'screen_size_inch', 'main_camera_mp']
    categorical_features = ['brand', 'has_5g']
    
    # Numeric Pipeline
    if model_type == 'linear':
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', StandardScaler())
        ])
    elif model_type == 'tree':
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median', add_indicator=True))
        ])
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'linear' or 'tree'.")
        
    # Categorical Pipeline
    # For brand and has_5g (which already contains "Yes", "No", "Unknown")
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any columns not explicitly specified
    )
    
    return preprocessor

if __name__ == "__main__":
    # Test building
    prep_linear = build_preprocessor('linear')
    prep_tree = build_preprocessor('tree')
    print("Linear preprocessor:", prep_linear)
    print("Tree preprocessor:", prep_tree)
