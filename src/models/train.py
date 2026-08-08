import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import sys

# Thêm đường dẫn để import được từ thư mục cha
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.features.pipeline import build_preprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_experiment_configs():
    return [
        {
            'name': 'Linear Regression (Raw)',
            'model': LinearRegression(),
            'model_type': 'linear',
            'target_log': False
        },
        {
            'name': 'Linear Regression (Log)',
            'model': LinearRegression(),
            'model_type': 'linear',
            'target_log': True
        },
        {
            'name': 'Ridge Regression (Raw)',
            'model': Ridge(alpha=1.0),
            'model_type': 'linear',
            'target_log': False
        },
        {
            'name': 'Ridge Regression (Log)',
            'model': Ridge(alpha=1.0),
            'model_type': 'linear',
            'target_log': True
        },
        {
            'name': 'Random Forest (Raw)',
            'model': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'model_type': 'tree',
            'target_log': False
        },
        {
            'name': 'Random Forest (Log)',
            'model': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
            'model_type': 'tree',
            'target_log': True
        }
    ]

def evaluate_pipeline(pipeline, X, y, target_log):
    preds = pipeline.predict(X)
    # Target values were kept as raw. If the pipeline itself uses TransformedTargetRegressor,
    # the predictions are already inverse transformed by sklearn!
    # Let's ensure this. TransformedTargetRegressor's predict() returns values in original space.
    # So we don't need manual expm1 here, UNLESS we manually transformed y outside.
    # We will pass y as raw VND.
    
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    return mae, rmse, r2

def main():
    data_dir = 'data/processed'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Load Data
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))['price_vnd']
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))['price_vnd']
    manifest = pd.read_csv(os.path.join(data_dir, 'split_manifest.csv'))
    
    # Extract groups matching the Train set
    train_manifest = manifest[manifest['split'] == 'train'].reset_index(drop=True)
    groups = train_manifest['model_family']
    
    gkf = GroupKFold(n_splits=5)
    
    # Dummy Baseline evaluation
    dummy_pipe = Pipeline(steps=[
        ('preprocessor', build_preprocessor('linear')),
        ('regressor', DummyRegressor(strategy='median'))
    ])
    
    dummy_maes = []
    for train_idx, val_idx in gkf.split(X_train, y_train, groups):
        dummy_pipe.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        mae, _, _ = evaluate_pipeline(dummy_pipe, X_train.iloc[val_idx], y_train.iloc[val_idx], False)
        dummy_maes.append(mae)
    logging.info(f"Dummy Baseline CV MAE: {np.mean(dummy_maes):.0f}")
    
    configs = get_experiment_configs()
    results = []
    
    for config in configs:
        maes, rmses, r2s = [], [], []
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            # Verify overlap
            g_tr = set(groups.iloc[train_idx])
            g_val = set(groups.iloc[val_idx])
            assert len(g_tr.intersection(g_val)) == 0, "DATA LEAKAGE: Group overlap in CV!"
            
            preprocessor = build_preprocessor(config['model_type'])
            
            if config['target_log']:
                regressor = TransformedTargetRegressor(
                    regressor=config['model'],
                    func=np.log1p,
                    inverse_func=np.expm1
                )
            else:
                regressor = config['model']
                
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', regressor)
            ])
            
            pipeline.fit(X_tr, y_tr)
            mae, rmse, r2 = evaluate_pipeline(pipeline, X_val, y_val, config['target_log'])
            maes.append(mae)
            rmses.append(rmse)
            r2s.append(r2)
            
        results.append({
            'Model': config['name'],
            'Target': 'Log' if config['target_log'] else 'Raw',
            'CV MAE': np.mean(maes),
            'CV RMSE': np.mean(rmses),
            'CV R²': np.mean(r2s),
            'Std MAE': np.std(maes),
            'config_ref': config
        })
        
    df_results = pd.DataFrame(results).sort_values(by='CV MAE')
    df_results_out = df_results.drop(columns=['config_ref'])
    df_results_out.to_csv(os.path.join(data_dir, 'model_comparison.csv'), index=False)
    logging.info("\n" + df_results_out.to_string())
    
    # Selection
    best_config = df_results.iloc[0]['config_ref']
    logging.info(f"Selected Best Model: {best_config['name']}")
    
    # Final Training on FULL TRAIN
    preprocessor = build_preprocessor(best_config['model_type'])
    if best_config['target_log']:
        regressor = TransformedTargetRegressor(
            regressor=best_config['model'],
            func=np.log1p,
            inverse_func=np.expm1
        )
    else:
        regressor = best_config['model']
        
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    final_pipeline.fit(X_train, y_train)
    joblib.dump(final_pipeline, os.path.join(models_dir, 'smartphone_price_model_v1.joblib'))
    logging.info("Model saved to models/smartphone_price_model_v1.joblib")
    
    # Evaluate on Test
    test_mae, test_rmse, test_r2 = evaluate_pipeline(final_pipeline, X_test, y_test, best_config['target_log'])
    logging.info(f"Final Test MAE: {test_mae:.0f}")
    logging.info(f"Final Test RMSE: {test_rmse:.0f}")
    logging.info(f"Final Test R²: {test_r2:.4f}")
    
    # Error Analysis
    test_preds = final_pipeline.predict(X_test)
    error_df = pd.DataFrame({
        'model_name': manifest[manifest['split'] == 'test'].reset_index(drop=True)['model_name'],
        'actual_price': y_test.values,
        'predicted_price': test_preds,
        'absolute_error': np.abs(y_test.values - test_preds)
    })
    error_df = error_df.sort_values(by='absolute_error', ascending=False)
    error_df.to_csv(os.path.join(data_dir, 'test_predictions.csv'), index=False)
    
    logging.info("Top 5 Prediction Errors on Test Set:")
    logging.info("\n" + error_df.head(5).to_string())

if __name__ == "__main__":
    main()
