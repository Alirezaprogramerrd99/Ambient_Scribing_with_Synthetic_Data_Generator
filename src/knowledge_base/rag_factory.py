"""
RAG Factory - Unified Interface for RAG System Creation

Provides a single entry point to create RAG systems using either
manual implementation or LlamaIndex, with seamless switching.

"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Dict, Any

from .retriever import (
    BaseRetriever,
    RetrievalResponse,
    create_retriever as create_manual_retriever,
)
from .indexer import (
    create_indexer as create_manual_indexer,
    KnowledgeBaseBuilder,
    EmbeddingManager,
)
from .document_processor import DocumentProcessor, DocumentType, ChunkingStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# RAG Backend Enum
# =============================================================================

class RAGBackend(str, Enum):
    """Available RAG backend implementations"""
    
    MANUAL = "manual"           # custom implementation
    LLAMA_INDEX = "llama_index" # LlamaIndex implementation
    HYBRID = "hybrid"           # LlamaIndex with custom medical processing
    
    @classmethod
    def from_string(cls, value: str) -> "RAGBackend":
        """Convert string to enum"""
        value_lower = value.lower().replace("-", "_")
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"Unknown RAG backend: {value}")


# =============================================================================
# RAG Configuration
# =============================================================================

class RAGConfig:
    """
    Configuration for RAG system
    
    Centralizes all RAG-related configuration options.
    """
    
    def __init__(
        self,
        backend: Union[str, RAGBackend] = RAGBackend.MANUAL,
        vector_store: str = "chroma",
        collection_name: str = "medical_knowledge",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        persist_dir: str = "./vector_store",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_top_k: int = 5,
        use_hybrid_search: bool = False,
        use_reranker: bool = False,
        use_query_expansion: bool = True,
        use_clinical_filtering: bool = True,
        # Qdrant-specific
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: Optional[str] = None,
    ):
        """
        Initialize RAG configuration
        
        Args:
            backend: RAG backend to use ("manual", "llama_index", "hybrid")
            vector_store: Vector store type ("chroma" or "qdrant")
            collection_name: Collection/index name
            embedding_model: HuggingFace embedding model
            persist_dir: Directory for persistence
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            similarity_top_k: Number of results to retrieve
            use_hybrid_search: Use hybrid (dense + sparse) search
            use_reranker: Use cross-encoder reranking
            use_query_expansion: Expand queries with medical knowledge
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            qdrant_api_key: Qdrant API key (for cloud)
        """
        self.backend = RAGBackend.from_string(backend) if isinstance(backend, str) else backend
        self.vector_store = vector_store
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.persist_dir = Path(persist_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_top_k = similarity_top_k
        self.use_hybrid_search = use_hybrid_search
        self.use_reranker = use_reranker
        self.use_query_expansion = use_query_expansion
        self.use_clinical_filtering = use_clinical_filtering
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "backend": self.backend.value,
            "vector_store": self.vector_store,
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model,
            "persist_dir": str(self.persist_dir),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "similarity_top_k": self.similarity_top_k,
            "use_hybrid_search": self.use_hybrid_search,
            "use_reranker": self.use_reranker,
            "use_query_expansion": self.use_query_expansion,
            "use_clinical_filtering": self.use_clinical_filtering,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConfig":
        """Create from dictionary"""
        return cls(**data)
    
    def __str__(self) -> str:
        return (
            f"RAGConfig(backend={self.backend.value}, "
            f"vector_store={self.vector_store}, "
            f"collection={self.collection_name})"
        )


# =============================================================================
# RAG System Factory
# =============================================================================

class RAGFactory:
    """
    Factory for creating RAG systems
    
    Provides unified interface to create retrievers using either
    manual implementation or LlamaIndex.
    
    Example:
        config = RAGConfig(backend="llama_index", vector_store="chroma")
        factory = RAGFactory(config)
        
        # Build knowledge base
        factory.build_knowledge_base("./medical_docs")
        
        # Get retriever
        retriever = factory.get_retriever()
        results = retriever.retrieve("chest pain guidelines")
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize factory
        
        Args:
            config: RAG configuration (uses defaults if not provided)
        """
        self.config = config or RAGConfig()
        self._retriever: Optional[BaseRetriever] = None
        
        logger.info(f"RAG Factory initialized: {self.config}")
    
    def build_knowledge_base(
        self,
        documents_dir: Union[str, Path],
        document_type: DocumentType = DocumentType.GENERAL,
        clear_existing: bool = False,
        use_semantic_chunking: bool = False,
    ) -> Dict[str, Any]:
        """
        Build knowledge base from documents
        
        Args:
            documents_dir: Path to documents directory
            document_type: Default document type
            clear_existing: Clear existing index first
            use_semantic_chunking: Use semantic chunking (LlamaIndex only)
            
        Returns:
            Dictionary with build statistics
        """
        documents_dir = Path(documents_dir)
        
        if not documents_dir.exists():
            raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
        
        logger.info(f"Building knowledge base from: {documents_dir}")
        logger.info(f"Backend: {self.config.backend.value}")
        
        if self.config.backend == RAGBackend.MANUAL:
            return self._build_manual_kb(documents_dir, document_type, clear_existing)
        else:
            return self._build_llama_index_kb(
                documents_dir, document_type, clear_existing, use_semantic_chunking
            )
    
    def _build_manual_kb(
        self,
        documents_dir: Path,
        document_type: DocumentType,
        clear_existing: bool,
    ) -> Dict[str, Any]:
        """Build knowledge base using manual implementation"""
        
        # Create indexer
        indexer = create_manual_indexer(
            vector_store=self.config.vector_store,
            embedding_model=self.config.embedding_model,
            collection_name=self.config.collection_name,
            persist_directory=str(self.config.persist_dir),
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            api_key=self.config.qdrant_api_key,
        )
        
        # Create processor
        processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
        )
        
        # Build
        builder = KnowledgeBaseBuilder(indexer, processor)
        stats = builder.build_from_directory(
            documents_dir,
            document_type=document_type,
            clear_existing=clear_existing,
        )
        
        return {
            "backend": "manual",
            "total_chunks": stats.total_chunks,
            "embedding_dimension": stats.embedding_dimension,
            "persist_dir": str(self.config.persist_dir),
        }
    
    def _build_llama_index_kb(
        self,
        documents_dir: Path,
        document_type: DocumentType,
        clear_existing: bool,
        use_semantic_chunking: bool,
    ) -> Dict[str, Any]:
        """Build knowledge base using LlamaIndex"""
        
        # Check if LlamaIndex is available
        try:
            from .llama_index_rag import LlamaIndexKnowledgeBaseBuilder, LlamaIndexSettings
        except ImportError as e:
            logger.error(f"LlamaIndex not available: {e}")
            raise ImportError(
                "LlamaIndex is required for this backend. Install with:\n"
                "pip install llama-index llama-index-embeddings-huggingface "
                "llama-index-vector-stores-chroma"
            )
        
        # Clear if requested
        if clear_existing and self.config.persist_dir.exists():
            import shutil
            shutil.rmtree(self.config.persist_dir)
            logger.info(f"Cleared existing index at {self.config.persist_dir}")
        
        # Create settings
        settings = LlamaIndexSettings(
            embedding_model=self.config.embedding_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        # Build
        builder = LlamaIndexKnowledgeBaseBuilder(
            settings=settings,
            vector_store=self.config.vector_store,
            persist_dir=str(self.config.persist_dir),
            collection_name=self.config.collection_name,
        )
        
        retriever = builder.build_from_directory(
            documents_dir,
            use_semantic_chunking=use_semantic_chunking,
        )
        
        # Cache the retriever
        self._retriever = retriever
        
        return {
            "backend": "llama_index",
            "persist_dir": str(self.config.persist_dir),
            "semantic_chunking": use_semantic_chunking,
        }
    
    def get_retriever(self) -> BaseRetriever:
        """
        Get configured retriever
        
        Returns:
            Retriever instance (manual or LlamaIndex based on config)
        """
        if self._retriever is not None:
            return self._retriever
        
        if self.config.backend == RAGBackend.MANUAL:
            self._retriever = self._create_manual_retriever()
        else:
            self._retriever = self._create_llama_index_retriever()
        
        return self._retriever
    
    def _create_manual_retriever(self) -> BaseRetriever:
        """Create manual retriever"""
        from .retriever import HybridRetriever
        
        base_retriever = create_manual_retriever(
            vector_store=self.config.vector_store,
            embedding_model=self.config.embedding_model,
            collection_name=self.config.collection_name,
            persist_directory=str(self.config.persist_dir),
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            api_key=self.config.qdrant_api_key,
        )
        
        if self.config.use_hybrid_search:
            return HybridRetriever(
                dense_retriever=base_retriever,
                sparse_weight=0.3,
                dense_weight=0.7,
            )
        
        return base_retriever
    
    def _create_llama_index_retriever(self) -> BaseRetriever:
        """Create LlamaIndex retriever"""
        try:
            from .llama_index_rag import (
                LlamaIndexRetriever,
                LlamaIndexIndexer,
                LlamaIndexSettings,
                HybridMedicalRetriever,
            )
        except ImportError:
            raise ImportError("LlamaIndex is required for this backend")
        
        # Create settings
        settings = LlamaIndexSettings(
            embedding_model=self.config.embedding_model,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        # Create indexer and load existing index
        indexer = LlamaIndexIndexer(
            settings=settings,
            vector_store=self.config.vector_store,
            persist_dir=str(self.config.persist_dir),
            collection_name=self.config.collection_name,
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            api_key=self.config.qdrant_api_key,
        )
        
        # Load existing index
        indexer.load_index()
        
        # Create base retriever.
        # When wrapped by HybridMedicalRetriever, the hybrid layer calls retrieve()
        # with top_k * 2 so the clinical filter has a meaningful pool to work with.
        # Set reranker_top_n accordingly so the reranker doesn't collapse those
        # extra candidates back to similarity_top_k before the hybrid layer sees them.
        use_hybrid = (
            self.config.backend == RAGBackend.HYBRID
            or self.config.use_query_expansion
            or self.config.use_clinical_filtering
        )
        reranker_top_n = (
            self.config.similarity_top_k * 2 if use_hybrid
            else self.config.similarity_top_k
        )
        base_retriever = LlamaIndexRetriever(
            indexer=indexer,
            similarity_top_k=self.config.similarity_top_k,
            use_reranker=self.config.use_reranker,
            reranker_top_n=reranker_top_n,
        )
        
        # Wrap with hybrid medical retriever if configured
        if use_hybrid:
            return HybridMedicalRetriever(
                llama_retriever=base_retriever,
                use_query_expansion=self.config.use_query_expansion,
                use_clinical_filtering=self.config.use_clinical_filtering,
            )
        
        return base_retriever


# =============================================================================
# Convenience Functions
# =============================================================================

def create_rag_system(
    backend: str = "manual",
    documents_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> BaseRetriever:
    """
    Create a complete RAG system with one function call
    
    Args:
        backend: RAG backend ("manual", "llama_index", "hybrid")
        documents_dir: Directory to index (if provided, builds KB first)
        **kwargs: Additional configuration options
        
    Returns:
        Configured retriever
        
    Example:
        # Quick setup with manual backend
        retriever = create_rag_system(
            backend="manual",
            documents_dir="./medical_knowledge",
            vector_store="chroma",
        )
        
        # Or with LlamaIndex
        retriever = create_rag_system(
            backend="llama_index",
            documents_dir="./medical_knowledge",
            use_semantic_chunking=True,
        )
    """
    # Create config
    config = RAGConfig(backend=backend, **{
        k: v for k, v in kwargs.items()
        if k in RAGConfig.__init__.__code__.co_varnames
    })
    
    # Create factory
    factory = RAGFactory(config)
    
    # Build KB if documents provided
    if documents_dir:
        factory.build_knowledge_base(
            documents_dir,
            use_semantic_chunking=kwargs.get("use_semantic_chunking", False),
            clear_existing=kwargs.get("clear_existing", False),
        )
    
    return factory.get_retriever()


def compare_backends(
    query: str,
    documents_dir: Union[str, Path],
    top_k: int = 5,
) -> Dict[str, RetrievalResponse]:
    """
    Compare retrieval results from different backends
    
    Useful for evaluation and thesis comparison.
    
    Args:
        query: Test query
        documents_dir: Documents directory
        top_k: Number of results
        
    Returns:
        Dictionary mapping backend name to RetrievalResponse
    """
    results = {}
    
    backends = [
        ("manual", RAGBackend.MANUAL),
        ("llama_index", RAGBackend.LLAMA_INDEX),
        ("hybrid", RAGBackend.HYBRID),
    ]
    
    for name, backend in backends:
        try:
            config = RAGConfig(
                backend=backend,
                persist_dir=f"./compare_{name}",
            )
            factory = RAGFactory(config)
            factory.build_knowledge_base(documents_dir, clear_existing=True)
            
            retriever = factory.get_retriever()
            response = retriever.retrieve(query, top_k=top_k)
            
            results[name] = response
            logger.info(f"{name}: {response.num_results} results, top_score={response.top_score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to test {name}: {e}")
            results[name] = None
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System Factory")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build knowledge base")
    build_parser.add_argument("documents_dir", type=str, help="Documents directory")
    build_parser.add_argument(
        "--backend", type=str, default="manual",
        choices=["manual", "llama_index", "hybrid"],
        help="RAG backend"
    )
    build_parser.add_argument(
        "--vector-store", type=str, default="chroma",
        choices=["chroma", "qdrant"],
        help="Vector store"
    )
    build_parser.add_argument("--persist-dir", type=str, default="./vector_store")
    build_parser.add_argument("--clear", action="store_true", help="Clear existing")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("query", type=str, help="Search query")
    query_parser.add_argument("--backend", type=str, default="manual")
    query_parser.add_argument("--persist-dir", type=str, default="./vector_store")
    query_parser.add_argument("--top-k", type=int, default=5)
    
    args = parser.parse_args()
    
    if args.command == "build":
        config = RAGConfig(
            backend=args.backend,
            vector_store=args.vector_store,
            persist_dir=args.persist_dir,
        )
        factory = RAGFactory(config)
        stats = factory.build_knowledge_base(
            args.documents_dir,
            clear_existing=args.clear,
        )
        print(f"\nKnowledge base built: {stats}")
    
    elif args.command == "query":
        config = RAGConfig(
            backend=args.backend,
            persist_dir=args.persist_dir,
        )
        factory = RAGFactory(config)
        retriever = factory.get_retriever()
        
        response = retriever.retrieve(args.query, top_k=args.top_k)
        
        print(f"\nQuery: {response.query}")
        print(f"Strategy: {response.strategy}")
        print(f"Time: {response.retrieval_time_ms:.2f}ms")
        print(f"\nResults ({response.num_results}):")
        
        for i, result in enumerate(response.results, 1):
            print(f"\n{i}. Score: {result.score:.3f}")
            print(f"   Source: {result.source_file}")
            print(f"   Text: {result.text[:200]}...")
    
    else:
        parser.print_help()




if __name__ == "__main__":
    main()