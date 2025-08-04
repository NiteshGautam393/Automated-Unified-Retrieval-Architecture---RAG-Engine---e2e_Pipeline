import os
import io
import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from minio import Minio
from dotenv import load_dotenv

# --- Load credentials from .env ---
load_dotenv("/opt/airflow/.env")
load_dotenv()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

RAW_BUCKET = "raw"
BRONZE_BUCKET = "bronze"
BRONZE_FILENAME = "bronze.parquet"

# --- Connect to MinIO ---
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# --- Create bronze bucket if it doesn't exist ---
if not minio_client.bucket_exists(BRONZE_BUCKET):
    minio_client.make_bucket(BRONZE_BUCKET)

# --- List raw objects ---
objects = minio_client.list_objects(RAW_BUCKET, recursive=True)
html_files = [obj.object_name for obj in objects if obj.object_name.endswith(".html")]

print(f"Found {len(html_files)} '.html' files in MinIO /{RAW_BUCKET}")

rows = []

# --- Helper function ---
def extract_p_text(soup, label):
    try:
        strong_tag = soup.find('strong', string=lambda t: t and label in t)
        if strong_tag:
            full_text = strong_tag.parent.get_text(strip=True)
            return full_text.replace(label, '').strip()
    except Exception:
        return ''
    return ''

# --- Process each file ---
for object_name in html_files:
    try:
        data = minio_client.get_object(RAW_BUCKET, object_name).read()

        html_start_index = data.find(b'<!DOCTYPE html>')
        if html_start_index == -1:
            print(f"Skipping {object_name} - no DOCTYPE found")
            continue

        html_content = data[html_start_index:]
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract fields
        article_title = soup.title.string.strip() if soup.title and soup.title.string else ''
        article_author = extract_p_text(soup, 'Author:')
        article_date = extract_p_text(soup, 'Published on:')
        article_description = extract_p_text(soup, 'Summary:')

        url_p_tag = soup.find('strong', string='URL:').parent if soup.find('strong', string='URL:') else None
        article_url = url_p_tag.find('a')['href'] if url_p_tag and url_p_tag.find('a') else ''

        article_body_div = soup.find('div', style=True)
        article_body = article_body_div.get_text(separator='\\n', strip=True) if article_body_div else ''

        dir_name = os.path.basename(os.path.dirname(object_name))
        parts = dir_name.replace('bbc_article_001_', '').replace('.html', '').rsplit('_', 2)
        safe_path = parts[0] if len(parts) == 3 else dir_name
        ts = f"{parts[1]}_{parts[2]}" if len(parts) == 3 else ''

        rows.append({
            'article_title': article_title,
            'article_author': article_author,
            'article_date': article_date,
            'status': 'ok' if article_body else 'missing body',
            'ingested_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'article_description': article_description,
            'article_url': article_url,
            'file_size': len(html_content),
            'article_body': article_body
        })

    except Exception as e:
        print(f"Failed to process {object_name}: {e}")

# --- Save to bronze.parquet ---
if rows:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df), buf)
    buf.seek(0)

    minio_client.put_object(
        BRONZE_BUCKET,
        BRONZE_FILENAME,
        buf,
        length=buf.getbuffer().nbytes,
        content_type="application/octet-stream"
    )

    print(f"Saved bronze.parquet to MinIO bucket '{BRONZE_BUCKET}' with {len(df)} rows")
else:
    print("No records to write to bronze.")