import os
import io
import uuid
import pandas as pd
import pyarrow as pa
import numpy as np
import tempfile
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from deltalake import write_deltalake
from minio import Minio
from dotenv import load_dotenv


@dataclass
class ProcessingStats:
    total_docs: int = 0
    successful_chunks: int = 0
    failed_docs: int = 0
    skipped_docs: int = 0

@dataclass
class ChunkFeatures:
    word_count: int
    char_count: int
    sentence_count: int
    has_numbers: bool
    has_quotes: bool
    has_questions: bool
    avg_word_length: float


class MinIOClient:
    def __init__(self):
        load_dotenv("/opt/airflow/.env")
        self.client = Minio(
            os.getenv("MINIO_ENDPOINT"),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=False
        )
        self.silver_bucket = "silver"
        self.gold_bucket = "gold"

        if not self.client.bucket_exists(self.gold_bucket):
            self.client.make_bucket(self.gold_bucket)

    def load_silver_data(self, filename: str = "silver_delta.parquet") -> Optional[pd.DataFrame]:
        try:
            response = self.client.get_object(self.silver_bucket, filename)
            return pd.read_parquet(io.BytesIO(response.read()))
        except Exception as e:
            print(f"Failed to load silver data: {e}")
            return None

    def save_gold_data(self, df: pd.DataFrame) -> bool:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                write_deltalake(temp_dir, pa.Table.from_pandas(df), mode="overwrite")

                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        local_path = os.path.join(root, file)
                        rel_path = os.path.relpath(local_path, temp_dir)
                        minio_path = f"delta/{rel_path.replace(os.sep, '/')}"
                        self.client.fput_object(self.gold_bucket, minio_path, local_path)

            buffer = io.BytesIO()
            df.to_parquet(buffer, index=False)
            buffer.seek(0)
            self.client.put_object(self.gold_bucket, "gold_chunks.parquet", buffer,
                                   length=buffer.getbuffer().nbytes)
            return True
        except Exception as e:
            print(f"Failed to save: {e}")
            return False


class TextChunker:
    def __init__(self, target_tokens: int = 200):
        self.target_chars = int(target_tokens * 4.5)

    def chunk_text(self, text: str) -> List[str]:
        if not isinstance(text, str) or len(text.split()) < 10:
            return []

        text = text.strip()
        if len(text) <= self.target_chars:
            return [text]

        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= self.target_chars:
                chunks.append(remaining.strip())
                break

            chunk_end = self._find_split_point(remaining)
            chunk = remaining[:chunk_end].strip()

            if len(chunk) > 30:
                chunks.append(chunk)

            remaining = remaining[chunk_end:].strip()

        return [c for c in chunks if len(c.strip()) > 30]

    def _find_split_point(self, text: str) -> int:
        start = max(0, self.target_chars - 200)
        end = min(len(text), self.target_chars + 200)
        region = text[start:end]

        # Try paragraph breaks
        para_break = region.find('\n\n')
        if para_break != -1:
            return start + para_break

        # Try sentence endings
        sentences = list(re.finditer(r'[.!?]\s+', region))
        if sentences:
            best = min(sentences, key=lambda m: abs((start + m.end()) - self.target_chars))
            return start + best.end()

        # Try word boundaries
        word_break = text.rfind(' ', self.target_chars - 100, self.target_chars + 50)
        if word_break != -1:
            return word_break

        return self.target_chars


class FeatureExtractor:
    @staticmethod
    def extract_features(text: str) -> ChunkFeatures:
        words = text.split()
        return ChunkFeatures(
            word_count=len(words),
            char_count=len(text),
            sentence_count=len(re.findall(r'[.!?]+', text)),
            has_numbers=bool(re.search(r'\b\d+\b', text)),
            has_quotes=bool(re.search(r'["\']', text)),
            has_questions='?' in text,
            avg_word_length=np.mean([len(word) for word in words]) if words else 0
        )


class GoldLayerProcessor:
    def __init__(self):
        self.minio_client = MinIOClient()
        self.chunker = TextChunker()
        self.extractor = FeatureExtractor()
        self.stats = ProcessingStats()

    def process(self):
        df = self.minio_client.load_silver_data()
        if df is None:
            return

        df = df[df['language'] == 'en'].drop_duplicates(subset=['id'])
        gold_data = []

        for _, row in df.iterrows():
            self.stats.total_docs += 1
            chunks = self._process_document(row)
            gold_data.extend(chunks)

        if not gold_data:
            print("No valid chunks created")
            return

        gold_df = pd.DataFrame(gold_data)
        print(f"Created {len(gold_df)} chunks from {gold_df['doc_id'].nunique()} documents")

        self.minio_client.save_gold_data(gold_df)

    def _process_document(self, row: pd.Series) -> List[Dict]:
        if not isinstance(row['text'], str) or len(row['text'].split()) < 20:
            self.stats.skipped_docs += 1
            return []

        try:
            chunks = self.chunker.chunk_text(row['text'])
            if not chunks:
                self.stats.failed_docs += 1
                return []

            doc_features = self.extractor.extract_features(row['text'])
            chunk_data = []

            for i, chunk_text in enumerate(chunks):
                if self._is_valid_chunk(chunk_text):
                    chunk_features = self.extractor.extract_features(chunk_text)
                    chunk_data.append({
                        'chunk_id': str(uuid.uuid4()),
                        'doc_id': row['id'],
                        'chunk_index': i,
                        'chunk_text': chunk_text,
                        'chunk_length': chunk_features.char_count,
                        'chunk_word_count': chunk_features.word_count,
                        'chunk_sentence_count': chunk_features.sentence_count,
                        'title': row['title'],
                        'url': row['url'],
                        'author': row.get('author', ''),
                        'date_published': row['date_published'],
                        'language': row['language'],
                        'source': row['source'],
                        'scrape_ts': row['scrape_ts'],
                        'doc_word_count': doc_features.word_count,
                        'total_chunks': len(chunks),
                        'has_numbers': chunk_features.has_numbers,
                        'has_quotes': chunk_features.has_quotes,
                        'has_questions': chunk_features.has_questions,
                        'avg_word_length': round(chunk_features.avg_word_length, 2)
                    })
                    self.stats.successful_chunks += 1

            return chunk_data

        except Exception:
            self.stats.failed_docs += 1
            return []

    def _is_valid_chunk(self, chunk: str) -> bool:
        if not chunk or len(chunk.strip()) < 30 or len(chunk.split()) < 10:
            return False
        text_chars = len(re.sub(r'[^a-zA-Z\s]', '', chunk))
        return text_chars / len(chunk) >= 0.5


def main():
    processor = GoldLayerProcessor()
    processor.process()


if __name__ == "__main__":
    main()