import os
import tempfile
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from minio import Minio
from dotenv import load_dotenv
import ollama

# Load environment variables from .env
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
CHROMA_BUCKET = "chroma"
CHROMA_COLLECTION_NAME = "gold_chunks"
MODEL_NAME = "all-MiniLM-L6-v2"
LLAMA_MODEL = "llama3:latest"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Local ChromaDB path (mounted from minio-data)
CHROMA_LOCAL_PATH = os.getenv("CHROMA_LOCAL_PATH", "/opt/airflow/minio-data/chroma/chroma_store")

# Global clients
chroma_client = None
collection = None
sentence_model = None
minio_client = None
ollama_client = None

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    max_results: Optional[int] = 5
    temperature: Optional[float] = 0.7

class SearchResult(BaseModel):
    chunk_text: str
    similarity_score: float
    metadata: Dict[str, Any]

class AnswerResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    chroma_loaded: bool
    model_loaded: bool
    llama_available: bool
    local_chroma_exists: bool

# Try to load ChromaDB from local path first, fallback to MinIO download
async def initialize_chroma():
    global chroma_client, collection, minio_client

    # Try local path first (faster)
    if os.path.exists(CHROMA_LOCAL_PATH):
        try:
            logger.info(f"Loading ChromaDB from local path: {CHROMA_LOCAL_PATH}")
            chroma_client = chromadb.PersistentClient(path=CHROMA_LOCAL_PATH)
            collections = chroma_client.list_collections()
            if collections:
                try:
                    collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
                    logger.info(f"Successfully loaded collection: {CHROMA_COLLECTION_NAME}")
                    return
                except:
                    collection = collections[0]
                    logger.info(f"Using first available collection: {collection.name}")
                    return
        except Exception as e:
            logger.warning(f"Failed to load from local path {CHROMA_LOCAL_PATH}: {e}")

    # Fallback to MinIO download
    logger.info("Local ChromaDB not found, downloading from MinIO...")
    try:
        chroma_temp_dir = await download_chroma_from_minio()
        
        # Try different subdirectories in downloaded content
        paths = [
            chroma_temp_dir,
            os.path.join(chroma_temp_dir, "chroma"),
            os.path.join(chroma_temp_dir, "chroma_store")
        ]

        for path in paths:
            if os.path.exists(path):
                try:
                    test_client = chromadb.PersistentClient(path=path)
                    collections = test_client.list_collections()
                    if collections:
                        chroma_client = test_client
                        try:
                            collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
                        except:
                            collection = collections[0]
                        logger.info("Successfully loaded ChromaDB from MinIO")
                        return
                except Exception as e:
                    logger.warning(f"Failed to load from {path}: {e}")
        
        raise Exception("Failed to initialize ChromaDB from MinIO download")
    except Exception as e:
        logger.error(f"Failed to download ChromaDB from MinIO: {e}")
        raise Exception("ChromaDB not available locally or from MinIO")

# Download from MinIO (fallback)
async def download_chroma_from_minio():
    global minio_client

    logger.info("Downloading ChromaDB files from MinIO...")
    chroma_temp_dir = tempfile.mkdtemp(prefix="chroma_")

    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    if not minio_client.bucket_exists(CHROMA_BUCKET):
        raise Exception(f"Bucket '{CHROMA_BUCKET}' not found")

    objects = list(minio_client.list_objects(CHROMA_BUCKET, recursive=True))
    if not objects:
        raise Exception(f"No ChromaDB files found in bucket '{CHROMA_BUCKET}'")

    for obj in objects:
        local_path = os.path.join(chroma_temp_dir, obj.object_name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        minio_client.fget_object(CHROMA_BUCKET, obj.object_name, local_path)

    logger.info("ChromaDB download complete")
    return chroma_temp_dir

# Initialize models
async def initialize_models():
    global sentence_model, ollama_client

    logger.info("Loading sentence transformer model...")
    sentence_model = SentenceTransformer(MODEL_NAME)
    
    logger.info("Connecting to Ollama...")
    try:
        ollama_client = ollama.Client(host=OLLAMA_BASE_URL)
        # Test connection with a simple ping
        response = ollama_client.generate(
            model=LLAMA_MODEL,
            prompt="Hello",
            options={"max_tokens": 5}
        )
        logger.info("Successfully connected to Ollama")
    except Exception as e:
        logger.warning(f"Ollama not available: {e}")
        ollama_client = None


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting RAG API initialization...")
        await initialize_chroma()
        await initialize_models()
        logger.info("RAG API initialization complete")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize RAG API: {e}")
        yield
    finally:
        pass

# App

app = FastAPI(
    title="RAG API",
    description="Semantic Search API with Ollama and ChromaDB",
    version="1.0",
    lifespan=lifespan,
    docs_url="/"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    print("health")
    return HealthResponse(
        status="running",
        chroma_loaded=chroma_client is not None and collection is not None,
        model_loaded=sentence_model is not None,
        llama_available=ollama_client is not None,
        local_chroma_exists=os.path.exists(CHROMA_LOCAL_PATH)
    )

@app.post("/search", response_model=List[SearchResult])
async def search(request: QuestionRequest):
    """Search for relevant chunks without LLM generation"""
    print("search")
    if not sentence_model or not collection:
        raise HTTPException(status_code=503, detail="Models not initialized")

    query_vec = sentence_model.encode([request.question])
    results = collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=request.max_results or 5,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    output = []
    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        score = 1.0 - distances[i] if i < len(distances) else 0.0
        output.append(SearchResult(chunk_text=doc, similarity_score=score, metadata=metadata))
    
    return output

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    print("ask")
    """Ask a question and get an LLM-generated answer with sources"""
    if not sentence_model or not collection:
        raise HTTPException(status_code=503, detail="ChromaDB or SentenceTransformer not initialized")

    start = time.time()
    
    # Get relevant chunks
    query_vec = sentence_model.encode([request.question])
    results = collection.query(
        query_embeddings=query_vec.tolist(),
        n_results=request.max_results or 5,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = results.get("documents", [[]])[0]
    context = "\n\n---\n\n".join(chunks)

    # Generate answer
    answer = context  # Fallback to context if Ollama fails
    if ollama_client:
        try:
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Answer directly and concisely using only the information in the context. 
            If the context doesn't contain enough information to answer the question, say so."""
            
            prompt = f"Context:\n{context}\n\nQuestion: {request.question}\n\nAnswer:"
            
            res = ollama_client.generate(
                model=LLAMA_MODEL,
                prompt=prompt,
                options={"temperature": request.temperature or 0.7, "num_predict": 800}
            )
            answer = res.get("response", "").strip()
            if not answer:
                answer = "I couldn't generate an answer. Please check the context."
        except Exception as e:
            logger.warning(f"Ollama generation failed, falling back to context: {e}")
            answer = f"Context available but LLM unavailable: {context[:500]}..."

    # Prepare sources
    duration = time.time() - start
    sources = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    for i, doc in enumerate(documents):
        metadata = metadatas[i] if i < len(metadatas) else {}
        score = 1.0 - distances[i] if i < len(distances) else 0.0
        sources.append(SearchResult(chunk_text=doc, similarity_score=score, metadata=metadata))

    return AnswerResponse(answer=answer, sources=sources, processing_time=duration)

@app.get("/stats")
async def stats():
    """Get statistics about the ChromaDB collection"""
    if not collection:
        raise HTTPException(status_code=503, detail="Collection not loaded")
    
    return {
        "collection_name": getattr(collection, "name", "unknown"),
        "total_chunks": collection.count(),
        "embedding_model": MODEL_NAME,
        "llm_model": LLAMA_MODEL,
        "chroma_local_path": CHROMA_LOCAL_PATH,
        "local_chroma_exists": os.path.exists(CHROMA_LOCAL_PATH),
        "ollama_available": ollama_client is not None
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
