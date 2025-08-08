# storage.py - Clean Database and Vector Operations

import asyncio
import time
import logging
import hashlib
import zlib
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, date
import asyncpg
import numpy as np
from sentence_transformers import SentenceTransformer
from chunking import FormChunk
from routing import RoutingResult
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCache:
    """Simple thread-safe cache without memory leaks"""
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return self._cache.get(key)
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = value
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

# Global caches
embedding_cache = SimpleCache(max_size=100)
query_cache = SimpleCache(max_size=200)

class ProductionDB:
    """Production database with CockroachDB support"""
    
    def __init__(self):
        self.pool = None
        self.provider = None
    
    async def initialize(self) -> bool:
        """Initialize database connection"""
        cockroach_url = os.getenv('COCKROACH_DATABASE_URL')
        
        if not cockroach_url:
            logger.error("COCKROACH_DATABASE_URL not found")
            return False
        
        try:
            self.pool = await asyncpg.create_pool(
                cockroach_url,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
            
            await self._test_connection()
            self.provider = "cockroach"
            logger.info("✅ Connected to CockroachDB")
            
            await self._create_schema()
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    async def _test_connection(self):
        """Test database connection"""
        async with self.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
    
    async def _create_schema(self):
        """Create database schema"""
        schema = """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id VARCHAR(255) PRIMARY KEY,
            form_type VARCHAR(50) NOT NULL,
            section_type VARCHAR(100) NOT NULL,
            section_number VARCHAR(50) NOT NULL,
            section_title TEXT NOT NULL,
            content TEXT NOT NULL,
            start_pos INTEGER NOT NULL,
            end_pos INTEGER NOT NULL,
            
            -- Company metadata
            cik VARCHAR(20) NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            filing_date DATE,
            fiscal_year INTEGER,
            fiscal_quarter INTEGER,
            
            -- Document metadata
            content_type VARCHAR(50),
            char_count INTEGER NOT NULL,
            filing_url TEXT,
            document_url TEXT,
            
            -- Attachment metadata
            parent_form_type VARCHAR(50),
            parent_filing_date DATE,
            attachment_number VARCHAR(50),
            attachment_description TEXT,
            is_attachment BOOLEAN DEFAULT FALSE,
            attachment_type VARCHAR(50),
            
            -- Vector storage
            embedding BYTEA,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Create indexes for efficient querying
        CREATE INDEX IF NOT EXISTS idx_ticker_form ON chunks(ticker, form_type, fiscal_year DESC);
        CREATE INDEX IF NOT EXISTS idx_sections ON chunks(form_type, section_type, section_number);
        CREATE INDEX IF NOT EXISTS idx_temporal ON chunks(fiscal_year DESC, fiscal_quarter DESC);
        CREATE INDEX IF NOT EXISTS idx_content_type ON chunks(content_type);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(schema)
        
        logger.info("Database schema created successfully")
    
    @asynccontextmanager
    async def get_connection(self):
        """Connection context manager"""
        async with self.pool.acquire() as conn:
            yield conn
    
    def _parse_date_safely(self, date_str) -> Optional[date]:
        """Parse date string safely"""
        if not date_str:
            return None
        
        if isinstance(date_str, str):
            try:
                if 'T' in date_str:
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00')).date()
                else:
                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S']:
                        try:
                            return datetime.strptime(date_str, fmt).date()
                        except ValueError:
                            continue
            except (ValueError, TypeError):
                pass
        
        return date_str if isinstance(date_str, date) else None
    
    async def store_chunks_batch(self, chunks: List[FormChunk], embeddings: List[bytes]) -> int:
        """Store chunks in batch"""
        if not chunks or len(chunks) != len(embeddings):
            return 0
        
        try:
            async with self.get_connection() as conn:
                for chunk, embedding in zip(chunks, embeddings):
                    # Parse dates safely
                    filing_date = self._parse_date_safely(chunk.filing_date)
                    parent_filing_date = self._parse_date_safely(chunk.parent_filing_date)
                    
                    await conn.execute("""
                        INSERT INTO chunks (
                            chunk_id, form_type, section_type, section_number, section_title,
                            content, start_pos, end_pos, cik, ticker, filing_date, fiscal_year,
                            fiscal_quarter, content_type, char_count, filing_url, document_url,
                            parent_form_type, parent_filing_date, attachment_number,
                            attachment_description, is_attachment, attachment_type, embedding
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            embedding = EXCLUDED.embedding,
                            created_at = NOW()
                    """,
                        chunk.chunk_id, chunk.form_type, chunk.section_type,
                        chunk.section_number, chunk.section_title, chunk.content,
                        chunk.start_pos, chunk.end_pos, chunk.cik, chunk.ticker,
                        filing_date, chunk.fiscal_year, chunk.fiscal_quarter,
                        chunk.content_type, chunk.char_count, chunk.filing_url,
                        chunk.document_url, chunk.parent_form_type, parent_filing_date,
                        chunk.attachment_number, chunk.attachment_description,
                        chunk.is_attachment, chunk.attachment_type, embedding
                    )
                
                return len(chunks)
                
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise
    
    async def search_with_routing(self, routing_result: RoutingResult, limit: int = 20) -> List[FormChunk]:
        """Search chunks using routing results with flexible matching"""
        conditions = ["1=1"]
        params = []
        
        # Ticker filtering
        if routing_result.recommended_tickers:
            conditions.append(f"ticker = ANY(${len(params) + 1})")
            params.append(routing_result.recommended_tickers)
        
        # Form type filtering
        if routing_result.recommended_forms:
            form_values = [f.value for f in routing_result.recommended_forms]
            conditions.append(f"form_type = ANY(${len(params) + 1})")
            params.append(form_values)
        
        # Section targeting - make this more flexible
        if routing_result.primary_targets:
            section_conditions = []
            for target in routing_result.primary_targets:
                # Use flexible matching instead of exact
                section_conditions.append(
                    f"(form_type = '{target.form_type.value}' AND section_type = '{target.section_type}' AND section_number LIKE '%{target.section_number}%')"
                )
                
                # Add content-based fallback
                if target.section_number == "1A":
                    section_conditions.append("(section_title ILIKE '%risk%' OR content_type = 'risk_factors')")
                elif target.section_number == "7":
                    section_conditions.append("(section_title ILIKE '%management%discussion%' OR content_type = 'mda')")
                elif target.section_number == "8":
                    section_conditions.append("(section_title ILIKE '%financial%statement%' OR content_type = 'financials')")
            
            if section_conditions:
                conditions.append(f"({' OR '.join(section_conditions)})")
        
        # Basic content filtering
        conditions.append("char_count > 100")
        
        query = f"""
            SELECT chunk_id, form_type, section_type, section_number, section_title,
                   content, start_pos, end_pos, cik, ticker, filing_date, fiscal_year,
                   fiscal_quarter, content_type, char_count, filing_url, document_url,
                   parent_form_type, parent_filing_date, attachment_number,
                   attachment_description, is_attachment, attachment_type, embedding
            FROM chunks
            WHERE {' AND '.join(conditions)}
            ORDER BY fiscal_year DESC NULLS LAST, char_count DESC
            LIMIT {limit}
        """
        
        chunks = []
        try:
            async with self.get_connection() as conn:
                rows = await conn.fetch(query, *params)
                
                for row in rows:
                    chunk = FormChunk(
                        form_type=row['form_type'],
                        section_type=row['section_type'],
                        section_number=row['section_number'],
                        section_title=row['section_title'],
                        content=row['content'],
                        start_pos=row['start_pos'],
                        end_pos=row['end_pos'],
                        cik=row['cik'],
                        ticker=row['ticker'],
                        filing_date=str(row['filing_date']) if row['filing_date'] else '',
                        fiscal_year=row['fiscal_year'],
                        fiscal_quarter=row['fiscal_quarter'],
                        chunk_id=row['chunk_id'],
                        content_type=row['content_type'],
                        char_count=row['char_count'],
                        filing_url=row['filing_url'],
                        document_url=row['document_url'],
                        parent_form_type=row['parent_form_type'],
                        parent_filing_date=str(row['parent_filing_date']) if row['parent_filing_date'] else None,
                        attachment_number=row['attachment_number'],
                        attachment_description=row['attachment_description'],
                        is_attachment=row['is_attachment'],
                        attachment_type=row['attachment_type'],
                        embedding=row['embedding']
                    )
                    chunks.append(chunk)
                    
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
        
        return chunks

class VectorStore:
    """Memory-efficient vector store with quantization"""
    
    def __init__(self):
        self.model = None
        self._device = "cpu"
    
    def _get_model(self) -> SentenceTransformer:
        """Lazy load model"""
        if self.model is None:
            self.model = SentenceTransformer('intfloat/e5-small-v2')
            self.model.eval()
        return self.model
    
    def _quantize_embedding(self, embedding: np.ndarray) -> bytes:
        """Quantize embedding for storage"""
        normalized = embedding / (np.linalg.norm(embedding) + 1e-8)
        quantized = (normalized * 127).astype(np.int8)
        return zlib.compress(quantized.tobytes(), level=6)
    
    def _dequantize_embedding(self, compressed_bytes: bytes) -> np.ndarray:
        """Restore quantized embedding"""
        quantized_bytes = zlib.decompress(compressed_bytes)
        quantized = np.frombuffer(quantized_bytes, dtype=np.int8)
        return quantized.astype(np.float32) / 127.0
    
    async def embed_texts_batch(self, texts: List[str]) -> List[bytes]:
        """Generate embeddings for batch of texts"""
        embeddings = []
        
        for text in texts:
            # Check cache first
            cache_key = hashlib.md5(text.encode()).hexdigest()
            cached = embedding_cache.get(cache_key)
            
            if cached:
                embeddings.append(cached)
            else:
                # Generate embedding
                model = self._get_model()
                text_truncated = text[:2048]  # Prevent memory issues
                
                try:
                    loop = asyncio.get_event_loop()
                    embedding = await loop.run_in_executor(
                        None,
                        lambda: model.encode([text_truncated], show_progress_bar=False)[0]
                    )
                    
                    quantized = self._quantize_embedding(embedding)
                    embedding_cache.put(cache_key, quantized)
                    embeddings.append(quantized)
                    
                except Exception as e:
                    logger.warning(f"Embedding failed: {e}")
                    # Fallback: zero embedding
                    zero_emb = np.zeros(384, dtype=np.float32)
                    quantized = self._quantize_embedding(zero_emb)
                    embeddings.append(quantized)
        
        return embeddings
    
    async def similarity_search(
        self, 
        query_text: str, 
        candidate_chunks: List[FormChunk], 
        top_k: int = 10
    ) -> List[Tuple[FormChunk, float]]:
        """Rank candidates by similarity to query"""
        if not candidate_chunks:
            return []
        
        # Get query embedding
        query_embeddings = await self.embed_texts_batch([query_text])
        query_vec = self._dequantize_embedding(query_embeddings[0])
        
        # Score candidates
        scored_chunks = []
        for chunk in candidate_chunks:
            try:
                # Use vector similarity if embedding available
                if chunk.embedding:
                    chunk_vec = self._dequantize_embedding(chunk.embedding)
                    cosine_sim = np.dot(query_vec, chunk_vec) / (
                        np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec) + 1e-8
                    )
                    score = float(cosine_sim)
                else:
                    # Fallback to keyword matching
                    query_words = set(query_text.lower().split())
                    content_words = set(chunk.content.lower().split())
                    overlap = len(query_words.intersection(content_words))
                    score = overlap / max(len(query_words), 1)
                
                # Apply content type boost
                if chunk.content_type in ['risk_factors', 'mda', 'financials']:
                    score *= 1.2
                
                scored_chunks.append((chunk, score))
                
            except Exception as e:
                logger.warning(f"Scoring failed for chunk {chunk.chunk_id}: {e}")
                scored_chunks.append((chunk, 0.1))  # Low score fallback
        
        # Sort by score and return top-k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]

class IngestionPipeline:
    """Document ingestion pipeline"""
    
    def __init__(self, db: ProductionDB, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store
    
    async def _is_filing_processed(self, filing_url: str) -> bool:
        """Check if filing is already processed"""
        async with self.db.get_connection() as conn:
            result = await conn.fetchval(
                "SELECT 1 FROM chunks WHERE filing_url = $1 LIMIT 1", 
                filing_url
            )
            return result is not None
    
    async def ingest_company(
        self, 
        ticker: str, 
        forms: List[str] = ["10-K", "10-Q"], 
        limit: int = 5
    ) -> Dict[str, Any]:
        """Ingest company filings"""
        from edgar import Company
        from utilities import ticker_to_cik, build_base_metadata
        from chunking import chunk_form, process_filing_attachments
        
        start_time = time.time()
        result = {
            "ticker": ticker,
            "chunks_processed": 0,
            "filings_processed": 0,
            "errors": []
        }
        
        try:
            cik = ticker_to_cik(ticker)
            if not cik:
                result["errors"].append(f"Could not find CIK for ticker {ticker}")
                return result
            
            company = Company(cik)
            filings = company.get_filings(form=forms).head(limit)
            
            logger.info(f"Processing {len(filings)} filings for {ticker}")
            
            for filing in filings:
                try:
                    # Skip if already processed
                    if await self._is_filing_processed(filing.url):
                        logger.info(f"Skipping already processed filing: {filing.form}")
                        continue
                    
                    # Get filing text
                    main_text = filing.text()
                    if len(main_text.strip()) < 100:
                        continue
                    
                    # Extract metadata
                    base_meta = build_base_metadata(filing, main_text)
                    
                    # Chunk main document
                    main_meta = base_meta.copy()
                    main_meta.pop('form_type', None)
                    main_chunks = chunk_form(main_text, filing.form, **main_meta)
                    
                    # Process attachments
                    attachment_chunks = process_filing_attachments(filing, base_meta)
                    
                    all_chunks = main_chunks + attachment_chunks
                    
                    # Generate embeddings
                    chunk_texts = [chunk.content[:2048] for chunk in all_chunks]
                    embeddings = await self.vector_store.embed_texts_batch(chunk_texts)
                    
                    # Store in database
                    stored_count = await self.db.store_chunks_batch(all_chunks, embeddings)
                    
                    result["chunks_processed"] += stored_count
                    result["filings_processed"] += 1
                    
                    logger.info(f"Processed {filing.form} for {ticker}: {stored_count} chunks")
                    
                except Exception as e:
                    error_msg = f"Failed to process {filing.form}: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.error(error_msg)
                    continue
            
        except Exception as e:
            error_msg = f"Failed to get filings for {ticker}: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
        
        result["processing_time"] = time.time() - start_time
        return result