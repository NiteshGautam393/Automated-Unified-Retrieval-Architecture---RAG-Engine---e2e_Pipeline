import os
import shutil
import tempfile
import pandas as pd
from deltalake import DeltaTable
from sentence_transformers import SentenceTransformer
import chromadb
from minio import Minio
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv("/opt/airflow/.env")
load_dotenv()

# Constants
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
GOLD_BUCKET = "gold"
CHROMA_BUCKET = "chroma"
DELTA_PATH_IN_MINIO = "delta/"
CHROMA_COLLECTION_NAME = "gold_chunks"

# Setup temp directories
tmp_dir = tempfile.mkdtemp(prefix="gold_to_chroma_")
delta_local_dir = os.path.join(tmp_dir, "gold_delta")
chroma_persist_dir = os.path.join(tmp_dir, "chroma_store")

try:
    # Initialize MinIO client
    minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)

    # Download Delta files from MinIO
    print(" Downloading Delta files...")
    os.makedirs(delta_local_dir, exist_ok=True)
    objects = minio_client.list_objects(GOLD_BUCKET, prefix=DELTA_PATH_IN_MINIO, recursive=True)
    for obj in objects:
        local_path = os.path.join(delta_local_dir, os.path.relpath(obj.object_name, DELTA_PATH_IN_MINIO))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        minio_client.fget_object(GOLD_BUCKET, obj.object_name, local_path)

    # Load Delta table and sentence transformer model
    print(" Loading Delta table...")
    dt = DeltaTable(delta_local_dir)
    df = dt.to_pandas()
    print(f"Loaded {len(df)} chunks")

    print("Loading sentence transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize ChromaDB
    print(" Initializing ChromaDB...")
    os.makedirs(chroma_persist_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    # Process and embed chunks in batches
    print(" Embedding and inserting into Chroma...")
    batch_size = 64
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        texts = batch["chunk_text"].tolist()
        ids = batch["chunk_id"].astype(str).tolist()

        # Prepare metadata
        metadatas = []
        for _, row in batch.iterrows():
            metadata = {
                "doc_id": str(row["doc_id"]),
                "title": str(row["title"]) if pd.notna(row["title"]) else "",
                "url": str(row["url"]) if pd.notna(row["url"]) else "",
                "date_published": str(row["date_published"]) if pd.notna(row["date_published"]) else "",
                "source": str(row["source"]) if pd.notna(row["source"]) else "",
                "chunk_index": int(row["chunk_index"]) if pd.notna(row["chunk_index"]) else 0,
                "chunk_word_count": int(row["chunk_word_count"]) if pd.notna(row["chunk_word_count"]) else 0
            }
            metadatas.append(metadata)

        # Embed and add to collection
        embeddings = model.encode(texts, show_progress_bar=False)
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        print(f" Processed batch {i // batch_size + 1}/{(len(df) + batch_size - 1) // batch_size}")

    # Wait for ChromaDB to persist files
    print(" Waiting for ChromaDB to persist files...")
    time.sleep(3)

    # Upload ChromaDB files to MinIO
    print(" Uploading Chroma DB to MinIO...")
    if not minio_client.bucket_exists(CHROMA_BUCKET):
        minio_client.make_bucket(CHROMA_BUCKET)

    uploaded_files = 0
    for root, dirs, files in os.walk(chroma_persist_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, chroma_persist_dir)
            minio_path = f"chroma_store/{rel_path.replace(os.sep, '/')}"

            try:
                minio_client.fput_object(CHROMA_BUCKET, minio_path, local_path)
                uploaded_files += 1
                print(f" Uploaded: {minio_path}")
            except Exception as e:
                print(f" Failed to upload {minio_path}: {e}")

    print(f" Upload complete. {uploaded_files} files uploaded to MinIO.")

except Exception as e:
    print(f" Error: {e}")
    import traceback

    traceback.print_exc()

finally:
    # Cleanup with file handle management
    try:
        if 'chroma_client' in locals():
            del chroma_client
        if 'collection' in locals():
            del collection
        time.sleep(1)

        if os.path.exists(tmp_dir):
            for attempt in range(3):
                try:
                    shutil.rmtree(tmp_dir)
                    print(f"Cleaned up temp files")
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        print(f" Manual cleanup needed: {tmp_dir}")
    except Exception as cleanup_error:
        print(f" Cleanup warning: {cleanup_error}")

print("Script completed!")