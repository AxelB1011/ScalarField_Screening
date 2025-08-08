# pipeline.py - Clean QA System Pipeline

import asyncio
import time
import logging
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
from google import genai
from storage import ProductionDB, VectorStore, IngestionPipeline, query_cache
from routing import Router
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    text: str
    provider: str
    success: bool = True

@dataclass
class QAResult:
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    routing_info: Dict[str, Any]
    provider_used: str
    processing_time: float

class LLMProvider:
    """LLM provider using Gemini"""
    
    def __init__(self):
        self.gemini_client = genai.Client()
    
    async def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using Gemini"""
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        if not gemini_key:
            return LLMResponse(
                text="No API key configured for LLM provider.",
                provider="error",
                success=False
            )
        
        try:
            # Extract messages by role
            system_msgs = [msg["content"] for msg in messages if msg["role"] == "system"]
            user_msgs = [msg["content"] for msg in messages if msg["role"] == "user"]
            
            # Combine content
            system_part = "\n".join(system_msgs) if system_msgs else ""
            user_part = "\n".join(user_msgs) if user_msgs else ""
            final_content = f"{system_part}\n\n{user_part}" if system_part else user_part
            
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_content
            )
            
            return LLMResponse(
                text=response.text,
                provider="gemini"
            )
            
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")
            return LLMResponse(
                text="I apologize, but I'm experiencing technical difficulties. Please try again later.",
                provider="fallback",
                success=False
            )

class QAEngine:
    """Clean QA engine with routing intelligence"""
    
    def __init__(self, db: ProductionDB, vector_store: VectorStore, llm: LLMProvider):
        self.db = db
        self.vector_store = vector_store
        self.llm = llm
        self.router = Router()
    
    async def _get_processed_filing_urls(self, ticker: str) -> set:
        """Get already processed filing URLs"""
        try:
            async with self.db.get_connection() as conn:
                result = await conn.fetch(
                    "SELECT DISTINCT filing_url FROM chunks WHERE ticker = $1",
                    ticker
                )
                return {row['filing_url'] for row in result}
        except Exception as e:
            logger.warning(f"Could not check processed filings for {ticker}: {e}")
            return set()
    
    async def smart_ingest_for_query(self, question: str) -> Dict[str, Any]:
        """Intelligently ingest data based on query needs"""
        routing_result = self.router.route_query(question)
        tickers = routing_result.recommended_tickers
        forms = None
        if routing_result.recommended_forms:
            forms = [f.value for f in routing_result.recommended_forms]
        else:
            forms = ["10-K", "10-Q", "3", "4", "5", "DEF 14A", "8-K"]
        
        logger.info(f"Router identified need for: {tickers}, {forms}")
        
        # Check existing data
        async with self.db.get_connection() as conn:
            existing_data = await conn.fetch("""
                SELECT ticker, fiscal_year, COUNT(*) as chunk_count
                FROM chunks
                WHERE ticker = ANY($1) AND form_type = ANY($2)
                GROUP BY ticker, fiscal_year
                ORDER BY fiscal_year DESC
            """, tickers, forms)
        
        # Determine if we need more data
        current_year = datetime.now().year
        needed_years = {current_year - 2, current_year - 1, current_year}
        
        total_new_chunks = 0
        ingestion = IngestionPipeline(self.db, self.vector_store)
        
        for ticker in tickers:
            try:
                existing_years = {row['fiscal_year'] for row in existing_data if row['ticker'] == ticker}
                missing_years = needed_years - existing_years
                
                if missing_years or not existing_years:
                    logger.info(f"Ingesting data for {ticker}")
                    result = await ingestion.ingest_company(ticker, forms, limit=3)
                    total_new_chunks += result.get('chunks_processed', 0)
                
            except Exception as e:
                logger.error(f"Error ingesting {ticker}: {e}")
                continue
        
        return {
            "new_chunks": total_new_chunks,
            "router_reasoning": routing_result.reasoning
        }
    
    async def ask(self, question: str, max_sources: int = 8) -> QAResult:
        """Process question and generate answer"""
        start_time = time.time()
        
        # Check cache
        cache_key = hashlib.md5(question.encode()).hexdigest()
        cached_result = query_cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for question: {question[:50]}...")
            return cached_result
        
        logger.info(f"Processing question: {question}")
        
        # Route the query
        routing_result = self.router.route_query(question)
        logger.info(f"Routing: {routing_result.reasoning}")
        
        if not routing_result.primary_targets:
            return QAResult(
                answer="I couldn't identify specific sections relevant to your question. Please try rephrasing.",
                sources=[],
                confidence=0.0,
                routing_info={"reasoning": routing_result.reasoning},
                provider_used="none",
                processing_time=time.time() - start_time
            )
        
        # Search for relevant chunks
        candidate_chunks = await self.db.search_with_routing(routing_result, limit=max_sources * 2)
        
        # If insufficient data, try smart ingestion
        if len(candidate_chunks) < max_sources // 2:
            logger.info("Insufficient data - using smart ingestion")
            ingest_result = await self.smart_ingest_for_query(question)
            logger.info(f"Smart ingestion: +{ingest_result['new_chunks']} chunks")
            
            # Retry search after ingestion
            candidate_chunks = await self.db.search_with_routing(routing_result, limit=max_sources * 2)
        
        if not candidate_chunks:
            return QAResult(
                answer="I couldn't find relevant information in the database for your question.",
                sources=[],
                confidence=0.3,
                routing_info={"reasoning": routing_result.reasoning},
                provider_used="none",
                processing_time=time.time() - start_time
            )
        
        logger.info(f"Found {len(candidate_chunks)} candidate chunks")
        
        # Rank by similarity
        ranked_results = await self.vector_store.similarity_search(
            question, candidate_chunks, max_sources
        )

        # print("RAANKED RESULTS: ", ranked_results)
        

        if not ranked_results:
            return QAResult(
                answer="I found some potentially relevant information but couldn't rank it properly.",
                sources=[],
                confidence=0.3,
                routing_info={"reasoning": routing_result.reasoning},
                provider_used="none",
                processing_time=time.time() - start_time
        )
        # Build context for LLM
        context_parts = []
        sources = []
        
        for i, (chunk, score) in enumerate(ranked_results, 1):
            # Enhanced context header
            context_header = f"[{i}] {chunk.ticker} {chunk.form_type}"
            if chunk.fiscal_year:
                context_header += f" (FY{chunk.fiscal_year}"
                if chunk.fiscal_quarter:
                    context_header += f" Q{chunk.fiscal_quarter}"
                context_header += ")"
            context_header += f" - {chunk.section_title}"
            
            if chunk.content_type:
                context_header += f" [{chunk.content_type}]"
            
            context_parts.append(f"{context_header}:\n{chunk.content}")
            
            # Source metadata
            sources.append({
                "id": str(i),
                "ticker": chunk.ticker,
                "form_type": chunk.form_type,
                "fiscal_year": chunk.fiscal_year,
                "fiscal_quarter": chunk.fiscal_quarter,
                "section": f"{chunk.section_type} {chunk.section_number}".strip(),
                "section_title": chunk.section_title,
                "filing_date": chunk.filing_date,
                "content_type": chunk.content_type,
                "document_url": chunk.document_url or chunk.filing_url,
                "is_attachment": chunk.is_attachment,
                "similarity_score": float(score),
                "char_count": chunk.char_count
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate LLM response
        messages = [{
            "role": "system",
            "content": """You are an expert financial analyst with deep SEC filing expertise.

Key instructions:
1. Provide precise, quantitative answers when possible
2. Always cite sources using [source 1], [source 2] format
3. If comparing companies, create clear side-by-side analysis
4. Highlight trends and key financial metrics with specific values
5. Flag any limitations in available data
6. Use professional financial terminology appropriately
7. Extract actual numbers from financial statements when available

Response format:
- Lead with key findings
- Support with specific data points
- Provide context about data sources (10-K vs 10-Q)
- End with comprehensive source citations"""
        }, {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer with analysis and citations:"
        }]
        
        try:
            llm_response = await self.llm.generate(messages)
            
            # Calculate confidence
            confidence = 0.8 if llm_response.success else 0.4
            if len(ranked_results) >= 3:
                confidence *= 1.1
            if routing_result.recommended_tickers:
                confidence *= 1.1
            confidence = min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            llm_response = LLMResponse(
                text="I found relevant information but couldn't generate a proper response due to technical issues.",
                provider="error",
                success=False
            )
            confidence = 0.3
        
        # Create final result
        result = QAResult(
            answer=llm_response.text,
            sources=sources,
            confidence=confidence,
            routing_info={
                "reasoning": routing_result.reasoning,
                "recommended_tickers": routing_result.recommended_tickers,
                "recommended_forms": [f.value for f in routing_result.recommended_forms],
                "candidate_count": len(candidate_chunks),
                "ranked_count": len(ranked_results)
            },
            provider_used=llm_response.provider,
            processing_time=time.time() - start_time
        )
        
        # Cache result
        query_cache.put(cache_key, result)
        logger.info(f"Generated answer using {llm_response.provider} in {result.processing_time:.1f}s")
        
        return result

class SECQASystem:
    """Main SEC QA System"""
    
    def __init__(self):
        self.db = ProductionDB()
        self.vector_store = VectorStore()
        self.llm = LLMProvider()
        self.ingestion = None
        self.qa = None
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize database
            if not await self.db.initialize():
                logger.error("Database initialization failed")
                return False
            
            # Initialize other components
            self.ingestion = IngestionPipeline(self.db, self.vector_store)
            self.qa = QAEngine(self.db, self.vector_store, self.llm)
            
            self.initialized = True
            logger.info("✅ SEC QA System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
    
    def _check_initialized(self):
        """Check if system is initialized"""
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
    
    async def ingest_companies(
        self, 
        tickers: List[str], 
        forms: List[str] = ["10-K", "10-Q"], 
        limit_per_company: int = 3
    ) -> Dict[str, Any]:
        """Ingest multiple companies"""
        self._check_initialized()
        results = {}
        
        for ticker in tickers:
            try:
                result = await self.ingestion.ingest_company(ticker, forms, limit_per_company)
                results[ticker] = result
                logger.info(f"Ingested {ticker}: {result['chunks_processed']} chunks")
                
                # Brief pause between companies
                await asyncio.sleep(0.5)
                
            except Exception as e:
                results[ticker] = {
                    "ticker": ticker,
                    "chunks_processed": 0,
                    "filings_processed": 0,
                    "errors": [f"Failed to process {ticker}: {str(e)}"],
                    "processing_time": 0
                }
        
        total_chunks = sum(r.get('chunks_processed', 0) for r in results.values())
        logger.info(f"✅ Batch ingestion complete: {total_chunks} total chunks")
        
        return results
    
    async def ask_question(self, question: str, max_sources: int = 8) -> QAResult:
        """Ask a question"""
        self._check_initialized()
        return await self.qa.ask(question, max_sources)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        self._check_initialized()
        
        try:
            async with self.db.get_connection() as conn:
                chunk_count = await conn.fetchval("SELECT COUNT(*) FROM chunks")
                ticker_count = await conn.fetchval("SELECT COUNT(DISTINCT ticker) FROM chunks")
                
                return {
                    "status": "healthy",
                    "database_provider": self.db.provider,
                    "chunks_stored": chunk_count,
                    "companies_indexed": ticker_count,
                    "cache_stats": {
                        "embedding_cache_size": len(query_cache._cache),
                        "query_cache_size": len(query_cache._cache)
                    },
                    "features": [
                        "Clean FormChunk chunking with full metadata",
                        "Multi-dimensional routing with context awareness", 
                        "Simple leak-free caching",
                        "Quantized embeddings for efficiency",
                        "Gemini LLM integration"
                    ]
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }