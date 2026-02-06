"""
RAG Retriever for Medical Knowledge Base

Handles retrieval of relevant medical knowledge for RAG-enhanced generation.
Supports dense, sparse, and hybrid retrieval strategies.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .indexer import EmbeddingManager

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RetrievalResult:
    """Single retrieval result with metadata"""
    
    text: str
    score: float
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def source_file(self) -> str:
        """Return a meaningful source identifier"""
        # Prefer title over raw file path, fall back to filename stem
        title = self.metadata.get("title")
        if title and title.strip():
            return title
        source = self.metadata.get("source_file", "")
        if source and source.strip():
            # Return just the filename, not the full path
            from pathlib import Path
            return Path(source).stem
        return "Unknown"
    
    @property
    def section_title(self) -> Optional[str]:
        return self.metadata.get("section_title")
    
    @property
    def document_type(self) -> str:
        return self.metadata.get("document_type", "general")
    
    def __str__(self) -> str:
        return f"RetrievalResult(score={self.score:.3f}, source={self.source_file})"


@dataclass
class RetrievalResponse:
    """Complete retrieval response with multiple results"""
    
    query: str
    results: List[RetrievalResult]
    retrieval_time_ms: float = 0.0
    strategy: str = "dense"
    
    @property
    def num_results(self) -> int:
        return len(self.results)
    
    @property
    def top_score(self) -> float:
        return self.results[0].score if self.results else 0.0
    
    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)
    
    def get_context(
        self,
        max_results: Optional[int] = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Get concatenated context from results
        
        Args:
            max_results: Maximum number of results to include
            separator: Separator between results
            
        Returns:
            Concatenated context string
        """
        results = self.results[:max_results] if max_results else self.results
        
        context_parts = []
        for i, result in enumerate(results, 1):
            source_info = f"[Source: {result.source_file}]"
            if result.section_title:
                source_info += f" [Section: {result.section_title}]"
            
            context_parts.append(f"{source_info}\n{result.text}")
        
        return separator.join(context_parts)
    
    def get_sources(self) -> List[str]:
        """Get list of unique source files"""
        return list(set(r.source_file for r in self.results))


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies"""
    DENSE = "dense"           # Semantic similarity only
    SPARSE = "sparse"         # BM25/keyword only
    HYBRID = "hybrid"         # Combined dense + sparse
    RERANKED = "reranked"     # Hybrid with cross-encoder reranking


# =============================================================================
# Abstract Base Retriever
# =============================================================================

class BaseRetriever(ABC):
    """Abstract base class for retrievers"""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResponse:
        """Retrieve relevant documents for a query"""
        pass
    
    @abstractmethod
    def retrieve_with_filter(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResponse:
        """Retrieve with metadata filters"""
        pass


# =============================================================================
# ChromaDB Retriever
# =============================================================================

class ChromaRetriever(BaseRetriever):
    """
    Retriever for ChromaDB vector store
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        collection_name: str = "medical_knowledge",
        persist_directory: Optional[str] = "./chroma_db",
    ):
        """
        Initialize ChromaDB retriever
        
        Args:
            embedding_manager: Embedding manager for query encoding
            collection_name: Name of the collection
            persist_directory: ChromaDB persistence directory
        """
        self.embedding_manager = embedding_manager
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            if self.persist_directory:
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                self._client = chromadb.Client(
                    Settings(anonymized_telemetry=False)
                )
            
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB retriever initialized: {self.collection_name}")
            
        except ImportError:
            raise ImportError("chromadb is required")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        **kwargs,
    ) -> RetrievalResponse:
        """
        Retrieve relevant documents using dense retrieval
        
        Args:
            query: Search query
            top_k: Number of results to return
            score_threshold: Minimum cosine similarity score (0.0-1.0).
                Results below this threshold are filtered out to avoid
                returning irrelevant content. Recommended: 0.35-0.5.
            
        Returns:
            RetrievalResponse with results
        """
        import time
        start_time = time.time()
        
        # Embed query
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Search — fetch extra results so we have enough after filtering
        fetch_k = max(top_k * 2, 10) if score_threshold > 0 else top_k
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )
        
        # Parse results
        retrieval_results = []
        
        if results and results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
            )):
                # Convert distance to similarity score (cosine distance to similarity)
                score = 1 - distance
                
                # Filter out low-relevance results
                if score < score_threshold:
                    logger.debug(
                        f"Filtered result {i} (score={score:.3f} < threshold={score_threshold})"
                    )
                    continue
                
                retrieval_results.append(RetrievalResult(
                    text=doc,
                    score=score,
                    chunk_id=metadata.get('chunk_id', f'chunk_{i}'),
                    metadata=metadata,
                ))
        
        # Trim to top_k after filtering
        retrieval_results = retrieval_results[:top_k]
        
        if not retrieval_results and score_threshold > 0:
            logger.warning(
                f"No results above score threshold {score_threshold} for query: "
                f"{query[:80]}..."
            )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=retrieval_results,
            retrieval_time_ms=elapsed_ms,
            strategy=RetrievalStrategy.DENSE.value,
        )
    
    def retrieve_with_filter(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResponse:
        """
        Retrieve with metadata filters
        
        Args:
            query: Search query
            filters: Metadata filters (e.g., {"document_type": "nice_guideline"})
            top_k: Number of results
            
        Returns:
            RetrievalResponse with filtered results
        """
        import time
        start_time = time.time()
        
        # Embed query
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Build ChromaDB where clause
        where_clause = {}
        for key, value in filters.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            else:
                where_clause[key] = value
        
        # Search with filter
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause if where_clause else None,
            include=["documents", "metadatas", "distances"],
        )
        
        # Parse results
        retrieval_results = []
        
        if results and results['documents'] and results['documents'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0],
            )):
                score = 1 - distance
                
                retrieval_results.append(RetrievalResult(
                    text=doc,
                    score=score,
                    chunk_id=metadata.get('chunk_id', f'chunk_{i}'),
                    metadata=metadata,
                ))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=retrieval_results,
            retrieval_time_ms=elapsed_ms,
            strategy=RetrievalStrategy.DENSE.value,
        )


# =============================================================================
# Qdrant Retriever
# =============================================================================

class QdrantRetriever(BaseRetriever):
    """
    Retriever for Qdrant vector store
    
    Supports advanced features like filtering and payload search.
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        collection_name: str = "medical_knowledge",
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant retriever
        
        Args:
            embedding_manager: Embedding manager
            collection_name: Collection name
            host: Qdrant host
            port: Qdrant port
            api_key: Optional API key
        """
        self.embedding_manager = embedding_manager
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.api_key = api_key
        self._client = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize Qdrant client"""
        try:
            from qdrant_client import QdrantClient
            
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
            )
            
            logger.info(f"Qdrant retriever initialized: {self.host}:{self.port}")
            
        except ImportError:
            raise ImportError("qdrant-client is required")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        **kwargs,
    ) -> RetrievalResponse:
        """
        Retrieve relevant documents
        
        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            RetrievalResponse with results
        """
        import time
        start_time = time.time()
        
        # Embed query
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Search — Qdrant supports native score_threshold
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_embedding,
            "limit": top_k,
            "with_payload": True,
        }
        if score_threshold > 0:
            search_params["score_threshold"] = score_threshold
        
        results = self._client.search(**search_params)
        
        # Parse results
        retrieval_results = []
        for result in results:
            retrieval_results.append(RetrievalResult(
                text=result.payload.get('text', ''),
                score=result.score,
                chunk_id=result.payload.get('chunk_id', str(result.id)),
                metadata=result.payload,
            ))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=retrieval_results,
            retrieval_time_ms=elapsed_ms,
            strategy=RetrievalStrategy.DENSE.value,
        )
    
    def retrieve_with_filter(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResponse:
        """
        Retrieve with metadata filters
        
        Args:
            query: Search query
            filters: Metadata filters
            top_k: Number of results
            
        Returns:
            RetrievalResponse with filtered results
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        import time
        start_time = time.time()
        
        # Embed query
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Build Qdrant filter
        conditions = []
        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )
        
        query_filter = Filter(must=conditions) if conditions else None
        
        # Search with filter
        results = self._client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        
        # Parse results
        retrieval_results = []
        for result in results:
            retrieval_results.append(RetrievalResult(
                text=result.payload.get('text', ''),
                score=result.score,
                chunk_id=result.payload.get('chunk_id', str(result.id)),
                metadata=result.payload,
            ))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=retrieval_results,
            retrieval_time_ms=elapsed_ms,
            strategy=RetrievalStrategy.DENSE.value,
        )


# =============================================================================
# Hybrid Retriever
# =============================================================================

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense and sparse retrieval
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from
    semantic search and keyword search.
    """
    
    def __init__(
        self,
        dense_retriever: BaseRetriever,
        sparse_weight: float = 0.3,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid retriever
        
        Args:
            dense_retriever: Dense (semantic) retriever
            sparse_weight: Weight for sparse results
            dense_weight: Weight for dense results
            rrf_k: RRF constant (default 60)
        """
        self.dense_retriever = dense_retriever
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
    
    def _sparse_search(
        self,
        query: str,
        documents: List[str],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Simple BM25-like sparse search
        
        Returns list of (doc_index, score) tuples
        """
        from collections import Counter
        import math
        
        # Tokenize query
        query_terms = query.lower().split()
        
        # Calculate IDF
        doc_count = len(documents)
        term_doc_freq = Counter()
        
        for doc in documents:
            doc_terms = set(doc.lower().split())
            for term in query_terms:
                if term in doc_terms:
                    term_doc_freq[term] += 1
        
        # Score documents
        scores = []
        for i, doc in enumerate(documents):
            doc_terms = doc.lower().split()
            doc_term_freq = Counter(doc_terms)
            
            score = 0.0
            for term in query_terms:
                if term in doc_term_freq:
                    tf = doc_term_freq[term]
                    df = term_doc_freq.get(term, 1)
                    idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1)
                    score += tf * idf
            
            scores.append((i, score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _rrf_fusion(
        self,
        dense_results: List[RetrievalResult],
        sparse_scores: List[Tuple[int, float]],
        all_docs: List[str],
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion
        """
        # Create score maps
        rrf_scores = {}
        
        # Add dense scores
        for rank, result in enumerate(dense_results):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + \
                self.dense_weight * (1 / (self.rrf_k + rank + 1))
        
        # Add sparse scores (we need to map indices back to chunk_ids)
        # This is a simplified version - in production you'd track this better
        for rank, (doc_idx, _) in enumerate(sparse_scores):
            # Guard against out-of-bounds index
            if doc_idx >= len(all_docs):
                continue
            # Find corresponding dense result by text matching
            for result in dense_results:
                if result.text == all_docs[doc_idx]:
                    chunk_id = result.chunk_id
                    rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + \
                        self.sparse_weight * (1 / (self.rrf_k + rank + 1))
                    break
        
        # Re-rank results
        result_map = {r.chunk_id: r for r in dense_results}
        
        fused_results = []
        for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            if chunk_id in result_map:
                result = result_map[chunk_id]
                fused_results.append(RetrievalResult(
                    text=result.text,
                    score=score,
                    chunk_id=chunk_id,
                    metadata=result.metadata,
                ))
        
        return fused_results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        **kwargs,
    ) -> RetrievalResponse:
        """
        Hybrid retrieval combining dense and sparse search
        
        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum similarity score — applied to final
                fused results rather than the dense retriever (since RRF
                produces different score magnitudes)
        """
        import time
        start_time = time.time()
        
        # Get dense results (fetch more for fusion; don't threshold here
        # because RRF rescores everything)
        dense_response = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        
        if not dense_response.results:
            return dense_response
        
        # Extract documents for sparse search
        documents = [r.text for r in dense_response.results]
        
        # Sparse search
        sparse_scores = self._sparse_search(query, documents, top_k * 2)
        
        # Fuse results
        fused_results = self._rrf_fusion(
            dense_response.results,
            sparse_scores,
            documents,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=fused_results[:top_k],
            retrieval_time_ms=elapsed_ms,
            strategy=RetrievalStrategy.HYBRID.value,
        )
    
    def retrieve_with_filter(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResponse:
        """Hybrid retrieval with filters"""
        # For filtered retrieval, we rely on the dense retriever's filtering
        # and apply sparse search on the filtered results
        
        import time
        start_time = time.time()
        
        dense_response = self.dense_retriever.retrieve_with_filter(
            query, filters, top_k=top_k * 2
        )
        
        if not dense_response.results:
            return dense_response
        
        documents = [r.text for r in dense_response.results]
        sparse_scores = self._sparse_search(query, documents, top_k * 2)
        
        fused_results = self._rrf_fusion(
            dense_response.results,
            sparse_scores,
            documents,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=fused_results[:top_k],
            retrieval_time_ms=elapsed_ms,
            strategy=RetrievalStrategy.HYBRID.value,
        )


# =============================================================================
# Medical Query Processor
# =============================================================================

class MedicalQueryProcessor:
    """
    Process and expand medical queries for better retrieval
    
    Handles medical abbreviations, synonyms, and query expansion.
    Supports both rule-based and LLM-based expansion.
    """
    
    # Common medical abbreviations
    ABBREVIATIONS = {
        "MI": "myocardial infarction heart attack",
        "CVA": "cerebrovascular accident stroke",
        "HTN": "hypertension high blood pressure",
        "DM": "diabetes mellitus",
        "T2DM": "type 2 diabetes mellitus",
        "COPD": "chronic obstructive pulmonary disease",
        "CHF": "congestive heart failure",
        "AF": "atrial fibrillation",
        "PE": "pulmonary embolism",
        "DVT": "deep vein thrombosis",
        "UTI": "urinary tract infection",
        "GERD": "gastroesophageal reflux disease",
        "OA": "osteoarthritis",
        "RA": "rheumatoid arthritis",
        "SOB": "shortness of breath dyspnea",
        "CP": "chest pain",
        "HA": "headache",
        "N/V": "nausea vomiting",
        "ACS": "acute coronary syndrome",
    }
    
    # Symptom synonyms
    SYNONYMS = {
        "chest pain": ["angina", "chest discomfort", "chest tightness"],
        "shortness of breath": ["dyspnea", "breathlessness", "difficulty breathing"],
        "headache": ["cephalalgia", "head pain"],
        "dizziness": ["vertigo", "lightheadedness"],
        "tiredness": ["fatigue", "lethargy", "weakness"],
        "stomach pain": ["abdominal pain", "epigastric pain", "belly pain"],
    }
    
    def __init__(self, prompt_manager=None):
        """
        Initialize query processor
        
        Args:
            prompt_manager: Optional PromptManager for LLM-based expansion
        """
        self.prompt_manager = prompt_manager
        self._llm_client = None
    
    def set_llm_client(self, llm_client):
        """
        Set LLM client for LLM-based query expansion
        
        Args:
            llm_client: Any object with a generate(prompt) or __call__(prompt) method
        """
        self._llm_client = llm_client
    
    @classmethod
    def expand_query(cls, query: str) -> str:
        """
        Expand query with medical synonyms and abbreviations (rule-based)
        
        Args:
            query: Original query
            
        Returns:
            Expanded query
        """
        expanded = query
        query_lower = query.lower()
        
        # Expand abbreviations
        for abbrev, expansion in cls.ABBREVIATIONS.items():
            if abbrev.lower() in query_lower or abbrev in query:
                expanded += f" {expansion}"
        
        # Add synonyms
        for term, synonyms in cls.SYNONYMS.items():
            if term in query_lower:
                expanded += " " + " ".join(synonyms)
        
        return expanded
    
    def expand_query_with_llm(
        self,
        query: str,
        use_rules_first: bool = True,
    ) -> str:
        """
        Expand query using LLM for better medical term coverage
        
        Combines rule-based expansion with LLM-based expansion for
        comprehensive query enrichment.
        
        Args:
            query: Original clinical query/scenario
            use_rules_first: Whether to apply rule-based expansion first
            
        Returns:
            Expanded query with medical terms, synonyms, and related concepts
        """
        # Step 1: Apply rule-based expansion first (fast, reliable)
        if use_rules_first:
            expanded = self.expand_query(query)
        else:
            expanded = query
        
        # Step 2: If no LLM client or prompt manager, return rule-based result
        if not self._llm_client or not self.prompt_manager:
            logger.debug("No LLM client configured, using rule-based expansion only")
            return expanded
        
        # Step 3: Use LLM to further expand the query
        try:
            prompt = self.prompt_manager.query_expansion(query=query)
            
            # Call LLM (support different client interfaces)
            if hasattr(self._llm_client, 'generate'):
                llm_response = self._llm_client.generate(
                    prompt,
                    temperature=0.3,  # Low temperature for consistency
                    max_tokens=200,
                )
            elif callable(self._llm_client):
                llm_response = self._llm_client(prompt)
            else:
                logger.warning("LLM client does not have generate method or is not callable")
                return expanded
            
            # Parse LLM response to extract expanded terms
            llm_expanded = self._parse_expansion_response(llm_response)
            
            # Combine rule-based and LLM-based expansions
            if llm_expanded:
                # Avoid duplicates by using set
                all_terms = set(expanded.lower().split())
                llm_terms = set(llm_expanded.lower().split())
                new_terms = llm_terms - all_terms
                
                if new_terms:
                    expanded += " " + " ".join(new_terms)
                    logger.debug(f"LLM added {len(new_terms)} new terms to query")
            
            return expanded
            
        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}. Using rule-based expansion.")
            return expanded
    
    def _parse_expansion_response(self, response: str) -> str:
        """
        Parse LLM response to extract expanded terms
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted expansion terms as space-separated string
        """
        import json
        import re
        
        response = response.strip()
        
        # Try to parse as JSON first
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                # Extract terms from various possible keys
                terms = []
                for key in ['expanded_terms', 'terms', 'synonyms', 'related_terms', 'keywords']:
                    if key in data:
                        value = data[key]
                        if isinstance(value, list):
                            terms.extend(value)
                        elif isinstance(value, str):
                            terms.append(value)
                
                if terms:
                    return " ".join(terms)
        except (json.JSONDecodeError, AttributeError):
            pass
        
        # Fallback: extract terms from plain text
        # Remove common filler words and punctuation
        cleaned = re.sub(r'[^\w\s]', ' ', response)
        cleaned = re.sub(r'\b(the|a|an|is|are|and|or|for|to|of|in|with)\b', '', cleaned.lower())
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        
        return cleaned
    
    @classmethod
    def extract_clinical_entities(cls, query: str) -> Dict[str, List[str]]:
        """
        Extract clinical entities from query
        
        Returns dict with symptoms, conditions, etc.
        """
        # This is a simple implementation
        # In production, use scispaCy or MedCAT
        
        entities = {
            "symptoms": [],
            "conditions": [],
            "body_parts": [],
        }
        
        query_lower = query.lower()
        
        # Common symptoms
        symptoms = [
            "pain", "ache", "discomfort", "swelling", "fever",
            "cough", "nausea", "vomiting", "diarrhea", "fatigue",
            "dizziness", "weakness", "numbness", "tingling",
        ]
        
        for symptom in symptoms:
            if symptom in query_lower:
                entities["symptoms"].append(symptom)
        
        # Body parts
        body_parts = [
            "chest", "head", "abdomen", "back", "leg", "arm",
            "neck", "shoulder", "knee", "ankle", "wrist",
        ]
        
        for part in body_parts:
            if part in query_lower:
                entities["body_parts"].append(part)
        
        return entities


# =============================================================================
# Factory Function
# =============================================================================

def create_retriever(
    vector_store: str = "chroma",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    collection_name: str = "medical_knowledge",
    use_hybrid: bool = False,
    **kwargs,
) -> BaseRetriever:
    """
    Factory function to create a retriever
    
    Args:
        vector_store: "chroma" or "qdrant"
        embedding_model: Embedding model name
        collection_name: Collection name
        use_hybrid: Whether to use hybrid retrieval
        **kwargs: Additional arguments
        
    Returns:
        Configured retriever instance
    """
    
    embedding_manager = EmbeddingManager(model_name=embedding_model)
    
    if vector_store.lower() == "chroma":
        base_retriever = ChromaRetriever(
            embedding_manager=embedding_manager,
            collection_name=collection_name,
            persist_directory=kwargs.get("persist_directory", "./chroma_db"),
        )
    elif vector_store.lower() == "qdrant":
        base_retriever = QdrantRetriever(
            embedding_manager=embedding_manager,
            collection_name=collection_name,
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 6333),
            api_key=kwargs.get("api_key"),
        )
    else:
        raise ValueError(f"Unknown vector store: {vector_store}")
    
    if use_hybrid:
        return HybridRetriever(
            dense_retriever=base_retriever,
            sparse_weight=kwargs.get("sparse_weight", 0.3),
            dense_weight=kwargs.get("dense_weight", 0.7),
        )
    
    return base_retriever


if __name__ == "__main__":
    # Test query processor
    print("Testing Medical Query Processor")
    print("=" * 60)
    
    processor = MedicalQueryProcessor()
    
    test_queries = [
        "patient with MI and SOB",
        "chest pain worse with exertion",
        "55 year old with HTN and T2DM",
    ]
    
    for query in test_queries:
        expanded = processor.expand_query(query)
        entities = processor.extract_clinical_entities(query)
        
        print(f"\nOriginal: {query}")
        print(f"Expanded: {expanded[:100]}...")
        print(f"Entities: {entities}")
    
    print("\n✓ Query processor tests passed!")