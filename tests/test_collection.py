import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/collection')))
from parser import parse_price, extract_variants_and_specs
from collect import parse_to_number

def test_parse_price():
    assert parse_price("17.000.000đ") == 17000000
    assert parse_price("17,000,000 VNĐ") == 17000000
    assert parse_price(" 17.000.000  ") == 17000000
    assert parse_price("Liên hệ") == None
    assert parse_price(None) == None

def test_parse_to_number():
    assert parse_to_number("8 GB") == 8
    assert parse_to_number("256 GB") == 256
    assert parse_to_number("5.000 mAh") == 5000
    assert parse_to_number(None) == None

def test_extract_variants_and_specs():
    sample_html = """
    <html>
        <body>
            <script type="application/ld+json">
            {
              "@context": "https://schema.org/",
              "@type": "Product",
              "name": "Samsung Galaxy S24 Ultra 5G",
              "brand": { "@type": "Brand", "name": "Samsung" },
              "offers": { "price": 24590000 },
              "additionalProperty": [
                { "name": "RAM", "value": "12 GB" },
                { "name": "ROM", "value": "256 GB" },
                { "name": "Pin", "value": "5000 mAh" }
              ]
            }
            </script>
        </body>
    </html>
    """
    variants = extract_variants_and_specs(sample_html)
    assert len(variants) == 1
    v = variants[0]
    assert v['brand'] == 'Samsung'
    assert 'S24 Ultra' in v['model_name']
    assert v['price_vnd'] == 24590000
    assert v['ram_raw'] == '12 GB'
    assert v['storage_raw'] == '256 GB'
    assert v['battery_raw'] == '5000 mAh'
    assert v['has_5g'] == 1

def test_extract_variants_apple_ram_bug():
    # Regression test for Apple RAM bug where storage (128GB) was taken as RAM
    sample_html = """
    <html>
        <body>
            <script type="application/ld+json">
            {
              "@context": "https://schema.org/",
              "@type": "Product",
              "name": "iPhone 16",
              "additionalProperty": [
                { "name": "Dung lượng RAM", "value": "8 GB" },
                { "name": "Bộ nhớ trong", "value": "128 GB" }
              ]
            }
            </script>
            <div>Điện thoại iPhone 16 256GB Chính hãng</div>
        </body>
    </html>
    """
    variants = extract_variants_and_specs(sample_html)
    v = variants[0]
    assert v['ram_raw'] == '8 GB'
    assert v['storage_raw'] == '128 GB'

    # Test fallback ignores storage if no RAM mentioned
    sample_html_no_ram = """
    <html>
        <body>
            <script type="application/ld+json">
            {
              "@context": "https://schema.org/",
              "@type": "Product",
              "name": "iPhone 16",
              "additionalProperty": [
                { "name": "Bộ nhớ trong", "value": "128 GB" }
              ]
            }
            </script>
            <div>Điện thoại iPhone 16 128GB Chính hãng</div>
        </body>
    </html>
    """
    variants_no_ram = extract_variants_and_specs(sample_html_no_ram)
    v2 = variants_no_ram[0]
    assert v2['ram_raw'] is None  # Should not mistakenly grab 128GB as RAM
    assert v2['storage_raw'] == '128 GB'

def test_extract_variants_mb_ram_bug():
    # Regression test for MB RAM
    sample_html = """
    <html>
        <body>
            <script type="application/ld+json">
            {
              "@context": "https://schema.org/",
              "@type": "Product",
              "name": "Nokia 105"
            }
            </script>
            <div>Điện thoại Nokia 105 64 MB RAM</div>
        </body>
    </html>
    """
    variants = extract_variants_and_specs(sample_html)
    v = variants[0]
    assert v['ram_raw'] == '64 MB'
