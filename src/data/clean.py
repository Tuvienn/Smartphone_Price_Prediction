import pandas as pd
import numpy as np
import re
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_brand(brand_raw):
    if pd.isna(brand_raw):
        return None
    b = str(brand_raw).strip().lower()
    
    canonical_mapping = {
        'oppo': 'Oppo',
        'redmagic': 'REDMAGIC',
        'apple': 'Apple',
        'samsung': 'Samsung',
        'xiaomi': 'Xiaomi',
        'honor': 'Honor',
        'tecno': 'Tecno',
        'nubia': 'Nubia',
        'nokia': 'Nokia',
        'benco': 'Benco',
        'itel': 'Itel',
        'inoi': 'Inoi',
        'tcl': 'TCL',
        'viettel': 'Viettel',
        'mobell': 'Mobell',
        'masstel': 'Masstel'
    }
    
    return canonical_mapping.get(b, str(brand_raw).strip().title())

def parse_ram(ram_raw):
    if pd.isna(ram_raw):
        return None
    match = re.search(r'(\d+)\s*(gb|mb|tb)', str(ram_raw).lower())
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit == 'mb':
            val = val / 1024.0
        elif unit == 'tb':
            val = val * 1024.0
        return val
    return None

def parse_storage(storage_raw):
    if pd.isna(storage_raw):
        return None
    match = re.search(r'(\d+)\s*(gb|mb|tb)', str(storage_raw).lower())
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        if unit == 'mb':
            val = val / 1024.0
        elif unit == 'tb':
            val = val * 1024.0
        return val
    return None

def parse_screen(screen_raw):
    if pd.isna(screen_raw):
        return None
    match = re.search(r'(\d+\.?\d*)', str(screen_raw))
    if match:
        return float(match.group(1))
    return None

def parse_camera(camera_raw):
    if pd.isna(camera_raw):
        return None
    # Find all floats/ints in the string and take max
    matches = re.findall(r'(\d+\.?\d*)', str(camera_raw))
    if matches:
        nums = [float(m) for m in matches]
        return max(nums)
    return None

def parse_5g(model_name):
    if pd.isna(model_name):
        return None
    name_lower = str(model_name).lower()
    if '5g' in name_lower:
        return 1
    elif '4g' in name_lower or 'lte' in name_lower:
        return 0
    return None

def normalize_model_name(name):
    if pd.isna(name):
        return ""
    # Lowercase, remove redundant spaces, remove GB variants (as they are captured in ram/storage)
    n = str(name).lower()
    n = re.sub(r'\s+', ' ', n)
    n = n.strip()
    return n

def run_cleaning():
    input_file = 'data/raw/fptshop_smartphones.csv'
    output_file = 'data/processed/fptshop_smartphones_clean.csv'
    
    if not os.path.exists(input_file):
        logging.error(f"Input file {input_file} not found.")
        return
        
    df = pd.read_csv(input_file)
    raw_rows = len(df)
    logging.info(f"Loaded {raw_rows} raw rows.")
    
    # 1. Duplicate Audit (Exact Row)
    df_exact_dupes = df[df.duplicated(keep=False)]
    exact_dupes_count = len(df) - len(df.drop_duplicates())
    if exact_dupes_count > 0:
        logging.info(f"Found {exact_dupes_count} exact duplicate rows. Removing them.")
        df = df.drop_duplicates()
        
    # Check duplicate URLs
    url_dupes = df[df.duplicated(subset=['product_url'], keep=False)]
    if not url_dupes.empty:
        logging.warning(f"Found {len(url_dupes)} rows with duplicated product_url. Flagged for audit.")
        
    # 2. Price Validation (Invalid Target)
    df['price_vnd'] = pd.to_numeric(df['price_vnd'], errors='coerce')
    invalid_price_mask = df['price_vnd'].isna() | (df['price_vnd'] <= 0)
    invalid_prices_count = invalid_price_mask.sum()
    if invalid_prices_count > 0:
        logging.info(f"Removing {invalid_prices_count} rows with invalid/missing/zero price.")
        df = df[~invalid_price_mask]
        
    # Flag abnormal prices
    abnormal_price_mask = (df['price_vnd'] < 1000000) | (df['price_vnd'] > 50000000)
    if abnormal_price_mask.sum() > 0:
        logging.warning(f"Flagged {abnormal_price_mask.sum()} rows with suspiciously high/low prices for review.")
        
    # 3. Categorical & Numeric Parsing
    df['brand'] = df['brand'].apply(normalize_brand)
    df['ram_gb'] = df['ram_raw'].apply(parse_ram)
    df['storage_gb'] = df['storage_raw'].apply(parse_storage)
    df['screen_size_inch'] = df['screen_size_inch'].apply(parse_screen)
    df['main_camera_mp'] = df['main_camera_mp'].apply(parse_camera)
    df['has_5g'] = df['model_name'].apply(parse_5g)
    
    # 4. Range Validation Flags
    ram_flag = (df['ram_gb'] > 32) | (df['ram_gb'] <= 0)
    if ram_flag.sum() > 0: logging.warning(f"Flagged {ram_flag.sum()} rows with abnormal RAM.")
    storage_flag = (df['storage_gb'] > 1024) | (df['storage_gb'] <= 0)
    if storage_flag.sum() > 0: logging.warning(f"Flagged {storage_flag.sum()} rows with abnormal Storage.")
    
    # 5. Semantic Duplicate Audit (model + RAM + storage + price)
    df['normalized_model_name'] = df['model_name'].apply(normalize_model_name)
    dedup_subset = ['brand', 'normalized_model_name', 'ram_gb', 'storage_gb', 'price_vnd']
    
    semantic_dupes_count = len(df) - len(df.drop_duplicates(subset=dedup_subset))
    if semantic_dupes_count > 0:
        logging.info(f"Removing {semantic_dupes_count} semantic duplicate variants (same model, specs, and price).")
        df = df.drop_duplicates(subset=dedup_subset)
        
    # Phase 4.1: Data Quality Correction (Apple & MB Units)
    suspicious_mask = (df['ram_gb'] > 32) | ((df['brand'] == 'Apple') & (df['ram_gb'] == df['storage_gb']))
    suspicious_count = suspicious_mask.sum()
    if suspicious_count > 0:
        logging.info(f"Phase 4.1: Found {suspicious_count} suspicious RAM rows. Fetching real RAM from source...")
        import sys
        sys.path.append(os.path.abspath('src/collection'))
        from parser import extract_variants_and_specs
        import requests
        
        corrected_count = 0
        nullified_count = 0
        
        for idx in df[suspicious_mask].index:
            url = df.loc[idx, 'product_url']
            try:
                res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                if res.status_code == 200:
                    variants = extract_variants_and_specs(res.text)
                    if variants:
                        real_ram_raw = variants[0]['ram_raw']
                        if real_ram_raw:
                            df.loc[idx, 'ram_gb'] = parse_ram(real_ram_raw)
                            corrected_count += 1
                        else:
                            df.loc[idx, 'ram_gb'] = None
                            nullified_count += 1
                    else:
                        df.loc[idx, 'ram_gb'] = None
                        nullified_count += 1
            except Exception as e:
                df.loc[idx, 'ram_gb'] = None
                nullified_count += 1
                
        logging.info(f"Phase 4.1: Corrected {corrected_count} rows, nullified {nullified_count} rows.")
        
    # 6. Missing Data Strategy
    # Drop rows missing RAM or Storage (Exclude Apple if RAM is missing per Phase 4.1 rule)
    missing_ram = df['ram_gb'].isna()
    missing_storage = df['storage_gb'].isna()
    drop_mask = missing_storage | (missing_ram & (df['brand'] != 'Apple'))
    
    missing_ram_rom_count = drop_mask.sum()
    if missing_ram_rom_count > 0:
        logging.info(f"Removing {missing_ram_rom_count} rows missing RAM (non-Apple) or Storage.")
        df = df[~drop_mask]
        
    # 7. Final Feature Selection
    # Drop Battery, Chipset, Refresh, Release Year (high missing)
    # Keep Camera (medium missing) as null
    features_to_keep = [
        'brand', 'model_name', 'ram_gb', 'storage_gb', 'screen_size_inch',
        'has_5g', 'main_camera_mp', 'price_vnd', 'product_url', 'collected_at'
    ]
    df_clean = df[features_to_keep]
    
    # Save Output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_clean.to_csv(output_file, index=False, encoding='utf-8')
    
    processed_rows = len(df_clean)
    rows_removed = raw_rows - processed_rows
    logging.info(f"Data Cleaning Complete!")
    logging.info(f"Raw rows: {raw_rows} -> Processed rows: {processed_rows} (Removed: {rows_removed})")
    logging.info(f"Saved to {output_file}")
    
if __name__ == "__main__":
    run_cleaning()
