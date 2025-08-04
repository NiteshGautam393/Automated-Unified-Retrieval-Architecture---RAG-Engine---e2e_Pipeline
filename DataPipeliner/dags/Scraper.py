import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urljoin, urlparse
import random
from datetime import datetime
import io
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
import os

load_dotenv("/opt/airflow/.env")
load_dotenv(".env")

print(" Debug: Loading environment variables...", flush=True)
print(f" Current working directory: {os.getcwd()}", flush=True)
print(f" Script location: {os.path.dirname(os.path.abspath(__file__))}", flush=True)

# Load credentials from .env
minio_endpoint = os.getenv("MINIO_ENDPOINT")
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")

if not all([minio_endpoint, minio_access_key, minio_secret_key]):
    print(" ERROR: Missing MinIO credentials!", flush=True)
    print(f"MINIO_ENDPOINT: {minio_endpoint}", flush=True)
    print(f"MINIO_ACCESS_KEY: {'***' if minio_access_key else 'NOT SET'}", flush=True)
    print(f"MINIO_SECRET_KEY: {'***' if minio_secret_key else 'NOT SET'}", flush=True)
    exit(1)
else:
    print(" MinIO credentials loaded successfully", flush=True)
    print(f" Endpoint: {minio_endpoint}", flush=True)


class BBCNewsScraper:
    def __init__(self):
        self.base_url = "https://aaa.bbb.ccc"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        self.scraped_urls = set()

        self.minio_client = Minio(
            minio_endpoint,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False
        )
        self.bucket_name = "raw"
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
                print(f" Created bucket: {self.bucket_name}")
            else:
                print(f" Bucket {self.bucket_name} already exists")
        except S3Error as e:
            print(f" Error creating/accessing bucket: {e}")

    def get_article_links(self, start_url="https://aaa.bbb.ccc/ddd"):
        article_links = []
        try:
            response = self.session.get(start_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            selectors = [
                'a[href*="/news/articles/"]',
                'a[href*="/news/world-"]',
                'a[href*="/news/uk-"]',
                'a[href*="/news/business-"]',
                'a[href*="/news/technology-"]',
                'a[href*="/news/science-"]',
                'a[href*="/news/health-"]',
            ]

            for sel in selectors:
                for link in soup.select(sel):
                    href = link.get('href')
                    if href:
                        # Ensure href is a string, not a list
                        if isinstance(href, list):
                            href = href[0] if href else None
                        
                        if href:  # Check again after potential list conversion
                            full_url = urljoin(self.base_url, href)
                            if full_url not in self.scraped_urls:
                                article_links.append(full_url)

        except Exception as e:
            print(f" Error getting links from {start_url}: {e}")
        return article_links

    def scrape_article(self, url):
        try:
            print(f" Scraping: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f" Error scraping {url}: {e}")
            return None

    def extract_title(self, soup):
        for sel in ['h1[data-testid="headline"]', 'h1', '.headline h1']:
            el = soup.select_one(sel)
            if el: return el.get_text().strip()
        return "No title found"

    def extract_author(self, soup):
        for sel in ['[data-testid="author-name"]', '.byline__name', '[rel="author"]']:
            el = soup.select_one(sel)
            if el: return el.get_text().strip()
        return "No author found"

    def extract_date(self, soup):
        for sel in ['time[datetime]', '[data-testid="timestamp"]', '.date']:
            el = soup.select_one(sel)
            if el:
                dt = el.get('datetime') or el.get_text()
                return dt.strip()
        return "No date found"

    def extract_summary(self, soup):
        for sel in ['meta[name="description"]', '[data-testid="summary"]']:
            el = soup.select_one(sel)
            if el:
                return el.get('content') or el.get_text().strip()
        return "No summary found"

    def extract_category(self, url):
        categories = {
            'world': 'World', 'uk': 'UK', 'business': 'Business',
            'technology': 'Technology', 'science': 'Science',
            'health': 'Health', 'sport': 'Sport'
        }
        for key, name in categories.items():
            if key in url: return name
        return "General"

    def extract_content(self, soup):
        content_parts = []
        for block in soup.select('[data-component="text-block"]'):
            text = block.get_text().strip()
            if len(text) > 40 and not any(j in text.lower() for j in ["whatsapp", "cookies", "share", "contact"]):
                content_parts.append(text)
        return '\n\n'.join(content_parts) if content_parts else "No content found"

    def extract_metadata(self, soup, url):
        return {
            "title": self.extract_title(soup),
            "author": self.extract_author(soup),
            "published_date": self.extract_date(soup),
            "summary": self.extract_summary(soup),
            "category": self.extract_category(url),
            "url": url
        }

    def build_clean_html(self, metadata, content):
        # Convert newlines to HTML paragraphs for better formatting
        formatted_content = content.replace('\n\n', '</p><p>').replace('\n', '<br>')
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{metadata['title']}</title>
</head>
<body>
  <h1>{metadata['title']}</h1>
  <p><strong>Author:</strong> {metadata['author']}</p>
  <p><strong>Published:</strong> {metadata['published_date']}</p>
  <p><strong>Category:</strong> {metadata['category']}</p>
  <p><strong>Summary:</strong> {metadata['summary']}</p>
  <p><strong>URL:</strong> <a href="{metadata['url']}">{metadata['url']}</a></p>
  <hr>
  <div style="line-height: 1.6; font-size: 16px;">
    <p>{formatted_content}</p>
  </div>
</body>
</html>'''

    def save_raw_article_to_minio(self, url, raw_response, index):
        try:
            soup = BeautifulSoup(raw_response.content, 'html.parser')
            metadata = self.extract_metadata(soup, url)
            content = self.extract_content(soup)

            if content == "No content found":
                print(" Skipped: No valid content.")
                return False

            html = self.build_clean_html(metadata, content).encode("utf-8")
            parsed = urlparse(url)
            safe_name = parsed.path.replace("/", "_").strip("_") or "index"
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bbc_article_{index:03d}_{safe_name}_{ts}.html"

            self.minio_client.put_object(
                self.bucket_name, filename, io.BytesIO(html),
                length=len(html), content_type="text/html"
            )
            print(f" Saved to MinIO: {filename}")
            return True

        except Exception as e:
            print(f" Error saving to MinIO: {e}")
            return False

    def scrape_multiple_articles(self, target_count=5):
        print(f"\n Starting to scrape {target_count} articles...\n")
        main_pages = [
            "https://aaa.bbb.ccc/news",
            "https://aaa.bbb.ccc/news/world",
            "https://aaa.bbb.ccc/news/uk",
            "https://aaa.bbb.ccc/news/business",
            "https://aaa.bbb.ccc/news/technology",
            "https://aaa.bbb.ccc/news/health"
        ]

        links = []
        for page in main_pages:
            links += self.get_article_links(page)
            time.sleep(random.uniform(1, 2))

        unique_links = list(set(links))[:target_count]
        print(f" Found {len(unique_links)} unique article URLs")

        count = 0
        for i, url in enumerate(unique_links, 1):
            if count >= target_count: break
            response = self.scrape_article(url)
            if response and self.save_raw_article_to_minio(url, response, count + 1):
                self.scraped_urls.add(url)
                count += 1
            time.sleep(random.uniform(1.5, 3))

        print(f"\n Done! Scraped {count} articles.\n")

    def print_summary(self):
        print(f"\n SCRAPING SUMMARY")
        print(f" Articles scraped: {len(self.scraped_urls)}")
        print(f" MinIO bucket: {self.bucket_name}")
        for i, url in enumerate(self.scraped_urls, 1):
            print(f"{i}. {url}")


def main():
    scraper = BBCNewsScraper()
    scraper.scrape_multiple_articles(target_count=10)
    scraper.print_summary()

if __name__ == "__main__":
    main()