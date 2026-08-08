import requests
from bs4 import BeautifulSoup
import time
import random
import logging

def get_product_urls(base_url="https://fptshop.com.vn/dien-thoai"):
    """
    Khám phá và lấy danh sách product URLs từ trang danh mục và các sub-category.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(base_url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching base category page: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Tìm các sub-category links (vd: /dien-thoai/samsung, /dien-thoai/apple)
    category_links = set([base_url])
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.startswith('/dien-thoai/') and len(href.split('/')) == 3:
            full_url = f"https://fptshop.com.vn{href}"
            category_links.add(full_url)
            
    logging.info(f"Found {len(category_links)} potential category/filter links.")
    
    product_urls = set()
    
    # Duyệt qua các category links để tìm product links
    for cat_url in category_links:
        try:
            time.sleep(random.uniform(1.0, 2.0))
            resp = requests.get(cat_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            cat_soup = BeautifulSoup(resp.text, 'html.parser')
            for a in cat_soup.find_all('a', href=True):
                href = a['href']
                if href.startswith('/dien-thoai/') and len(href.split('/')) == 3:
                    full_url = f"https://fptshop.com.vn{href}"
                    product_urls.add(full_url)
        except Exception as e:
            logging.error(f"Error fetching {cat_url}: {e}")
            
    return list(product_urls)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    urls = get_product_urls()
    print(f"Found {len(urls)} URLs.")

