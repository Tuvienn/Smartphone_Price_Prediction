import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def map_5g(val):
    if pd.isna(val):
        return "Unknown"
    elif val == 1.0 or val == 1:
        return "Yes"
    elif val == 0.0 or val == 0:
        return "No"
    else:
        return "Unknown"

def make_dataset():
    input_path = 'data/processed/fptshop_smartphones_clean.csv'
    output_dir = 'data/processed'
    
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return
        
    df = pd.read_csv(input_path)
    logging.info(f"Loaded {len(df)} rows from {input_path}")
    
    # 1. Ánh xạ biến has_5g sang dạng phân loại theo đúng Plan
    df['has_5g'] = df['has_5g'].apply(map_5g)
    
    # Đảm bảo target không bị null
    df = df.dropna(subset=['price_vnd'])
    
    # 2. Định nghĩa Features
    # Metadata và high-cardinality bị loại khỏi X
    metadata_cols = ['model_name', 'product_url', 'collected_at']
    target_col = 'price_vnd'
    
    X = df.drop(columns=[target_col] + metadata_cols)
    y = df[target_col]
    
    import re
    def get_family(name):
        n = str(name).lower()
        n = re.sub(r'\b\d+\s*(gb|mb|tb)\b', '', n)
        n = re.sub(r'\b(chính hãng|sim viettel|5g|4g)\b', '', n)
        n = re.sub(r'\s+', ' ', n).strip()
        return n
        
    df['model_family'] = df['model_name'].apply(get_family)
    groups = df['model_family']
    
    # 3. Train/Test Split với GroupShuffleSplit (80% Train, 20% Test)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    
    # 4. Save Split Manifest
    manifest = pd.DataFrame({
        'model_name': df['model_name'],
        'model_family': df['model_family'],
        'split': ['train' if i in train_idx else 'test' for i in range(len(df))]
    })
    
    # Check Leakage
    train_groups = set(manifest[manifest['split'] == 'train']['model_family'])
    test_groups = set(manifest[manifest['split'] == 'test']['model_family'])
    overlap = train_groups.intersection(test_groups)
    
    if len(overlap) > 0:
        logging.error(f"DATA LEAKAGE DETECTED! Overlapping groups: {overlap}")
    else:
        logging.info("Zero leakage: Train Groups ∩ Test Groups = 0")
        
    # 5. Lưu ra file csv
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    manifest.to_csv(os.path.join(output_dir, 'split_manifest.csv'), index=False)
    
    logging.info(f"Train/Test split complete!")
    logging.info(f"Train set: {len(X_train)} rows")
    logging.info(f"Test set: {len(X_test)} rows")
    logging.info(f"Unique groups in Train: {len(train_groups)}")
    logging.info(f"Unique groups in Test: {len(test_groups)}")
    logging.info(f"Group overlap count: {len(overlap)}")
    logging.info(f"GroupKFold strategy will be applied during Phase 6 training using model_family.")
    
if __name__ == "__main__":
    make_dataset()
