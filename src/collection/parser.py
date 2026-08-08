import re
import json
from bs4 import BeautifulSoup


def parse_price(price_str):
    if not price_str:
        return None
    price_str = price_str.lower().replace('vnđ', '').replace('đ', '').replace('.', '').replace(',', '').strip()
    try:
        return int(price_str)
    except ValueError:
        return None

def extract_variants_and_specs(html):
    """
    Phân tích HTML trang chi tiết sản phẩm lấy từ JSON-LD SEO.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    variant = {
        'brand': 'Unknown',
        'model_name': 'Unknown',
        'ram_raw': None,
        'storage_raw': None,
        'chipset': None,
        'battery_raw': None,
        'screen_size_inch': None,
        'refresh_rate_hz': None,
        'main_camera_mp': None,
        'front_camera_mp': None,
        'has_5g': 0,
        'release_year': None,
        'price_vnd': None
    }
    
    # Tìm script JSON-LD Product
    scripts = soup.find_all('script', type='application/ld+json')
    product_data = None
    for script in scripts:
        content = script.string
        if content:
            try:
                data = json.loads(content)
                if data.get('@type') == 'Product':
                    product_data = data
                    break
            except Exception:
                continue
                
    if not product_data:
        # Không phải trang sản phẩm hợp lệ hoặc lỗi
        return []
        
    variant['model_name'] = product_data.get('name', 'Unknown')
    
    if '5g' in variant['model_name'].lower():
        variant['has_5g'] = 1
        
    brand_info = product_data.get('brand', {})
    if isinstance(brand_info, dict) and 'name' in brand_info:
        variant['brand'] = brand_info['name']
        
    offers = product_data.get('offers', {})
    if isinstance(offers, dict) and 'price' in offers:
        try:
            variant['price_vnd'] = int(offers['price'])
        except (ValueError, TypeError):
            pass
            
    # Lấy Specs từ additionalProperty
    additional_props = product_data.get('additionalProperty', [])
    if isinstance(additional_props, list):
        for prop in additional_props:
            name = prop.get('name', '').lower()
            value = prop.get('value', '')
            if not value: continue
            
            if 'ram' in name:
                variant['ram_raw'] = value
            elif name == 'rom' or name == 'bộ nhớ trong':
                variant['storage_raw'] = value
            elif 'kích thước màn hình' in name:
                variant['screen_size_inch'] = value
            elif 'camera' in name:
                variant['main_camera_mp'] = value
            elif 'pin' in name or 'dung lượng' in name:
                variant['battery_raw'] = value
                
    # Fallback cho storage_raw nếu JSON-LD thiếu ROM nhưng có trong text HTML
    if not variant['storage_raw']:
        text_content = soup.get_text(separator=' | ')
        storage_match = re.search(r'(64|128|256|512)\s*GB|1\s*TB', text_content)
        if storage_match:
            variant['storage_raw'] = storage_match.group(0).strip()
            
    # Fallback cho ram_raw nếu JSON-LD thiếu RAM
    if not variant['ram_raw']:
        text_content = soup.get_text(separator=' | ')
        ram_match = re.search(r'(?i)(\d+)\s*(GB|MB)\s*RAM', text_content)
        if ram_match:
            variant['ram_raw'] = f"{ram_match.group(1)} {ram_match.group(2).upper()}"
            
    return [variant]
