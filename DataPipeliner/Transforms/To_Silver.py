import os
import io
import hashlib
import uuid
import pandas as pd
import pyarrow as pa
import duckdb
import re
import unicodedata
from deltalake import write_deltalake
from bs4 import BeautifulSoup
from minio import Minio
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from textstat import flesch_reading_ease
import warnings

# Set seed for langdetect to ensure consistent results
DetectorFactory.seed = 0

# --- Load credentials from .env ---
load_dotenv("/opt/airflow/.env")
load_dotenv()
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

# --- Configuration ---
RAW_BUCKET = "raw"
BRONZE_BUCKET = "bronze"
SILVER_BUCKET = "silver"
BRONZE_FILENAME = "bronze.parquet"
SILVER_FILENAME = "silver_delta.parquet"

# --- Connect to MinIO ---
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

# Create silver bucket if it doesn't exist
if not minio_client.bucket_exists(SILVER_BUCKET):
    minio_client.make_bucket(SILVER_BUCKET)
    print(f"Created MinIO bucket: {SILVER_BUCKET}")


def extract_main_content(html_content):
    """Extract main article content using multiple strategies"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Strategy 1: Look for semantic HTML5 elements
    main_content = soup.find('article') or soup.find('main') or soup.find('[role="main"]')

    # Strategy 2: Look for common article containers
    if not main_content:
        main_content = soup.find('div', class_=re.compile(r'(article|content|post|story)', re.I))

    # Strategy 3: Content density analysis - find div with most text
    if not main_content:
        all_divs = soup.find_all('div')
        if all_divs:
            main_content = max(all_divs, key=lambda d: len(d.get_text(strip=True)))

    return main_content or soup


def advanced_content_processing(article_body):
    """Multi-step content processing with enhanced cleaning"""
    # Extract main content
    main_content = extract_main_content(article_body)

    # Remove unwanted elements
    for tag in main_content(
            ['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'a', 'img', 'figure']):
        tag.decompose()

    # Extract paragraphs separately to maintain structure
    paragraphs = [p.get_text(strip=True) for p in main_content.find_all('p') if p.get_text(strip=True)]
    full_text = '\n\n'.join(paragraphs) if paragraphs else main_content.get_text(separator=' ', strip=True)

    # Advanced text normalization
    full_text = unicodedata.normalize('NFKD', full_text)
    full_text = re.sub(r'\s+', ' ', full_text)  # Normalize whitespace
    full_text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', full_text)  # Clean special chars

    return full_text.strip(), len(paragraphs)


def content_quality_filter(text, title, date):
    """Enhanced content quality checks"""
    # Word count check
    word_count = len(text.split())
    if word_count < 100 or word_count > 10000:
        return False, "Word count out of range"

    # Title-content relevance (basic check)
    if title:
        title_words = set(title.lower().split())
        text_words = set(text.lower().split()[:100])  # First 100 words
        overlap = len(title_words.intersection(text_words))
        if overlap < 2:  # Minimum title relevance
            return False, "Low title relevance"

    # Date validation
    try:
        pd.to_datetime(date)
    except:
        return False, "Invalid date"

    # Content coherence check (basic)
    if text.count('.') < word_count / 50:  # Too few sentences
        return False, "Low sentence density"

    return True, "Passed"


def enhanced_language_detection(text):
    """Enhanced language detection with confidence scoring"""
    try:
        clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
        if len(clean_text) < 20:
            return 'unknown'

        language_code = detect(clean_text)

        # Additional validation for English
        if language_code == 'en':
            try:
                reading_score = flesch_reading_ease(clean_text[:1000])  # First 1000 chars
                if reading_score < 10:  # Very difficult text, might be non-English
                    return 'unknown'
            except:
                pass  # If textstat fails, continue with original detection

        return language_code

    except LangDetectException:
        return 'unknown'
    except Exception:
        return 'unknown'


def extract_content_features(text, title):
    """Extract additional content features"""
    word_count = len(text.split())
    char_count = len(text)
    paragraph_count = text.count('\n\n') + 1

    # Reading time estimate (average 200 words per minute)
    reading_time_minutes = word_count / 200

    # Topic indicators (simple keyword-based)
    topics = []
    text_lower = text.lower()
    if any(word in text_lower for word in ['election', 'vote', 'politician', 'government']):
        topics.append('politics')
    if any(word in text_lower for word in ['economy', 'market', 'business', 'financial']):
        topics.append('business')
    if any(word in text_lower for word in ['health', 'medical', 'hospital', 'doctor']):
        topics.append('health')
    if any(word in text_lower for word in ['sport', 'team', 'match', 'player']):
        topics.append('sports')
    if any(word in text_lower for word in ['technology', 'digital', 'computer', 'internet']):
        topics.append('technology')

    # Content quality indicators
    has_numbers = bool(re.search(r'\d+', text))
    has_quotes = bool('"' in text or "'" in text)
    question_count = text.count('?')

    return {
        'word_count': word_count,
        'char_count': char_count,
        'paragraph_count': paragraph_count,
        'reading_time_minutes': round(reading_time_minutes, 2),
        'topics': ','.join(topics) if topics else 'general',
        'has_numbers': has_numbers,
        'has_quotes': has_quotes,
        'question_count': question_count
    }


# Load Bronze Data from MinIO
try:
    print(f"Reading bronze.parquet from MinIO bucket '{BRONZE_BUCKET}'...")

    response = minio_client.get_object(BRONZE_BUCKET, BRONZE_FILENAME)
    parquet_data = io.BytesIO(response.read())

    temp_bronze_path = 'temp_bronze.parquet'
    with open(temp_bronze_path, 'wb') as f:
        parquet_data.seek(0)
        f.write(parquet_data.read())

    con = duckdb.connect()
    df = con.execute(f"""
        SELECT *
        FROM read_parquet('{temp_bronze_path}')
        WHERE length(article_body) > 50  -- Filter out empty or very short articles
        AND status = 'ok'  -- Only process articles with valid body content
    """).df()
    con.close()

    os.remove(temp_bronze_path)

    print(f"Loaded {len(df)} rows from MinIO bronze bucket.")

except Exception as e:
    print(f"Failed to read bronze data from MinIO: {e}")
    print("Make sure the bronze.parquet file exists in the bronze bucket.")
    exit(1)

# Enhanced Silver Transformations
silver_rows = []
processed_count = 0
quality_stats = {
    'total_processed': 0,
    'quality_passed': 0,
    'quality_failed': 0,
    'language_english': 0,
    'language_other': 0,
    'failure_reasons': {}
}

print("Starting enhanced content processing...")

for _, row in df.iterrows():
    processed_count += 1
    quality_stats['total_processed'] += 1

    if processed_count % 100 == 0:
        print(f"Processed {processed_count} articles...")

    # Advanced content processing
    text, paragraph_count = advanced_content_processing(row['article_body'])

    # If the text is still empty after cleaning, skip
    if not text.strip():
        quality_stats['quality_failed'] += 1
        quality_stats['failure_reasons']['empty_text'] = quality_stats['failure_reasons'].get('empty_text', 0) + 1
        continue

    # Enhanced language detection
    language = enhanced_language_detection(text)

    if language != 'en':
        quality_stats['language_other'] += 1
        continue

    quality_stats['language_english'] += 1

    # Content quality filtering
    quality_passed, quality_reason = content_quality_filter(text, row['article_title'], row['article_date'])

    if not quality_passed:
        quality_stats['quality_failed'] += 1
        quality_stats['failure_reasons'][quality_reason] = quality_stats['failure_reasons'].get(quality_reason, 0) + 1
        continue

    quality_stats['quality_passed'] += 1

    # Extract enhanced features
    content_features = extract_content_features(text, row['article_title'])

    # Create a content hash for deduplication
    content_hash = hashlib.md5((row['article_title'] + text[:500]).encode('utf-8')).hexdigest()

    silver_rows.append({
        'id': str(uuid.uuid4()),
        'url': row['article_url'],
        'title': row['article_title'],
        'author': row['article_author'],
        'date_published': row['article_date'],
        'text': text,
        'language': language,
        'scrape_ts': row['ingested_at'],
        'source': 'bbc',
        'content_hash': content_hash,
        'description': row['article_description'],
        'file_size': row['file_size'],
        'paragraph_count': paragraph_count,
        **content_features  # Unpack all content features
    })

silver_df = pd.DataFrame(silver_rows)
print(f"Processed {len(silver_df)} high-quality English articles from {len(df)} total articles.")

# Enhanced deduplication
initial_count = len(silver_df)
silver_df = silver_df.drop_duplicates(subset=['content_hash'], keep='first')
duplicates_removed = initial_count - len(silver_df)
if duplicates_removed > 0:
    print(f"Deduplicated {duplicates_removed} rows. Final count: {len(silver_df)}")

# Enhanced QC Checks and Data Quality Statistics
if not silver_df.empty:
    total_rows = len(silver_df)
    missing_title = silver_df['title'].eq('').sum()
    missing_text = silver_df['text'].eq('').sum()
    missing_date = silver_df['date_published'].eq('').sum()

    print("\n" + "=" * 50)
    print("ENHANCED SILVER QC/DATA QUALITY REPORT")
    print("=" * 50)

    print(f"\n PROCESSING SUMMARY:")
    print(f"Total articles processed: {quality_stats['total_processed']}")
    print(f"Quality checks passed: {quality_stats['quality_passed']}")
    print(f"Quality checks failed: {quality_stats['quality_failed']}")
    print(f"English articles: {quality_stats['language_english']}")
    print(f"Non-English articles: {quality_stats['language_other']}")

    print(f"\n FAILURE REASONS:")
    for reason, count in quality_stats['failure_reasons'].items():
        print(f"  {reason}: {count} articles")

    print(f"\n FINAL DATASET STATISTICS:")
    print(f"Total rows: {total_rows}")
    print(f"Rows missing title: {missing_title} ({(missing_title / total_rows):.2%})")
    print(f"Rows missing text: {missing_text} ({(missing_text / total_rows):.2%})")
    print(f"Rows missing date: {missing_date} ({(missing_date / total_rows):.2%})")

    print(f"\n CONTENT FEATURES:")
    print(f"Average word count: {silver_df['word_count'].mean():.0f}")
    print(f"Average reading time: {silver_df['reading_time_minutes'].mean():.1f} minutes")
    print(f"Average paragraphs: {silver_df['paragraph_count'].mean():.1f}")

    print(f"\n TOPIC DISTRIBUTION:")
    topic_counts = {}
    for topics_str in silver_df['topics']:
        for topic in topics_str.split(','):
            topic = topic.strip()
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {count} articles")

    # Save to MinIO Silver Bucket
    try:
        print(f"\n Saving enhanced silver data to MinIO bucket '{SILVER_BUCKET}'...")

        # Drop the temporary content_hash column before saving
        final_df = silver_df.drop(columns=['content_hash'])

        # Convert to parquet format in memory
        buf = io.BytesIO()
        final_df.to_parquet(buf, index=False)
        buf.seek(0)

        # Upload to MinIO silver bucket
        minio_client.put_object(
            SILVER_BUCKET,
            SILVER_FILENAME,
            buf,
            length=buf.getbuffer().nbytes,
            content_type="application/octet-stream"
        )

        print(f" Silver data successfully saved to MinIO bucket '{SILVER_BUCKET}' as '{SILVER_FILENAME}'")

        # Also save as CSV for easy viewing
        csv_filename = "silver_data.csv"
        csv_buf = io.BytesIO()
        final_df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)

        minio_client.put_object(
            SILVER_BUCKET,
            csv_filename,
            csv_buf,
            length=csv_buf.getbuffer().nbytes,
            content_type="text/csv"
        )

        print(f" Also saved as CSV: '{csv_filename}' for easy viewing")

    except Exception as e:
        print(f" Failed to save silver data to MinIO: {e}")

else:
    print("\n No high-quality English articles found to create the Silver table.")

print(f"\n Enhanced Silver transformation complete!")
print(f" Data pipeline: Raw Bucket → Bronze Bucket → Enhanced Silver Bucket")
print(f" Check your MinIO console to view the enhanced silver data files.")
print(f" Total articles ready for semantic chunking: {len(silver_df)}")
