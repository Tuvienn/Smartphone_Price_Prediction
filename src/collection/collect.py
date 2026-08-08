import time
import random
import logging
import pandas as pd
from datetime import datetime
import requests

from discover import get_product_urls
from parser import extract_variants_and_specs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_to_number(raw_str):
    if not raw_str:
        return None
    try:
        # Extract digits
        digits = ''.join(filter(str.isdigit, raw_str))
        if digits:
            return int(digits)
    except Exception:
        pass
    return None

def run_collection():
    logging.info("Starting Full Data Collection (Phase 2.3)")
    
    # 1. Lấy danh sách URL
    urls = get_product_urls()
    logging.info(f"Discovered {len(urls)} URLs for full collection.")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    
    collected_data = []
    seen_identifiers = set()
    failed_urls = []
    duplicate_count = 0
    
    for url in urls:
        logging.info(f"Fetching: {url}")
        
        try:
            # Random delay 1-2s
            time.sleep(random.uniform(1.0, 2.5))
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            variants = extract_variants_and_specs(response.text)
            
            for variant in variants:
                # Tạo cột parsed
                ram_gb_parsed = parse_to_number(variant.get('ram_raw'))
                storage_gb_parsed = parse_to_number(variant.get('storage_raw'))
                
                # Tránh duplicate bằng identifier: normalized model + ram + storage
                model = variant.get('model_name', 'Unknown')
                ident = f"{model.lower().strip()}_{ram_gb_parsed}_{storage_gb_parsed}"
                
                if ident in seen_identifiers:
                    logging.info(f"Duplicate variant skipped: {ident}")
                    duplicate_count += 1
                    continue
                seen_identifiers.add(ident)
                
                # Bổ sung metadata
                variant['ram_gb_parsed'] = ram_gb_parsed
                variant['storage_gb_parsed'] = storage_gb_parsed
                variant['product_url'] = url
                variant['collected_at'] = datetime.now().isoformat()
                
                collected_data.append(variant)
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")
            failed_urls.append(url)
            continue
        except Exception as e:
            logging.error(f"Unexpected error parsing {url}: {e}")
            failed_urls.append(url)
            continue

    logging.info(f"Finished collection. Total variants collected: {len(collected_data)}. Failed URLs: {len(failed_urls)}. Duplicates skipped: {duplicate_count}.")
    
    if failed_urls:
        logging.warning(f"Failed URLs list: {failed_urls}")

    # Export to CSV
    if collected_data:
        df = pd.DataFrame(collected_data)
        # Đảm bảo đúng thứ tự column theo Schema
        cols_order = [
            'brand', 'model_name', 'ram_raw', 'ram_gb_parsed', 'storage_raw', 'storage_gb_parsed', 
            'chipset', 'battery_raw', 'screen_size_inch', 'refresh_rate_hz', 
            'main_camera_mp', 'front_camera_mp', 'has_5g', 'release_year', 
            'price_vnd', 'product_url', 'collected_at'
        ]
        
        # Thêm các cột còn thiếu nếu parser chưa trả ra
        for col in cols_order:
            if col not in df.columns:
                df[col] = None
                
        df = df[cols_order]
        output_path = "data/raw/fptshop_smartphones.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        logging.info(f"Data saved to {output_path}")
    else:
        logging.error("No data collected to save.")

if __name__ == "__main__":
    run_collection()
