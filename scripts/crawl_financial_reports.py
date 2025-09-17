# mbb_report_crawler.py (Scrapy + Selenium version)

import os
import re
import scrapy
from scrapy.selector import Selector
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import requests
from tqdm import tqdm

BASE_URL = "https://s.cafef.vn/bao-cao-tai-chinh/mbb.chn"
DOWNLOAD_DIR = "./reports/MBB"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0 Safari/537.36"
}

def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver

def fetch_html_with_selenium(url):
    driver = setup_driver()
    driver.get(url)

    buttons = driver.find_elements("css selector", ".tshowhide")
    for btn in buttons:
        try:
            driver.execute_script("arguments[0].click();", btn)
        except:
            pass

    html = driver.page_source
    driver.quit()
    return html

def extract_pdf_links(html, base_url):
    sel = Selector(text=html)
    links = []
    for a in sel.css("a::attr(href)").getall():
        if a.endswith(".pdf"):
            full_url = urljoin(base_url, a)
            text = os.path.basename(a)
            links.append((text, full_url))
    return links

def download_file(name, url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    filename = re.sub(r'[^\w\-_\. ]', '_', name) + ".pdf"
    dest_path = os.path.join(dest_folder, filename)

    try:
        with requests.get(url, headers=HEADERS, stream=True, timeout=15) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return dest_path
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def main():
    print("Fetching page with Selenium...")
    html = fetch_html_with_selenium(BASE_URL)

    print("Extracting report links...")
    pdf_links = extract_pdf_links(html, BASE_URL)
    print(f"Found {len(pdf_links)} PDF links")

    for name, url in tqdm(pdf_links, desc="Downloading reports"):
        saved_file = download_file(name, url, DOWNLOAD_DIR)
        if saved_file:
            print(f"Downloaded: {saved_file}")

if __name__ == "__main__":
    main()
