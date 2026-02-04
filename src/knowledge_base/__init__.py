"""
Knowledge Base Package

Document processing, indexing, and retrieval for RAG-enhanced generation.

Supports two RAG backends:
1. Manual Implementation - Custom code with full control
2. LlamaIndex Implementation - Industry-standard framework

Use RAGFactory to switch between backends seamlessly.

Example:
    from src.knowledge_base import RAGFactory, RAGConfig, RAGBackend
    
    # Manual backend
    config = RAGConfig(backend=RAGBackend.MANUAL)
    factory = RAGFactory(config)
    factory.build_knowledge_base("./medical_docs")
    retriever = factory.get_retriever()
    
    # Or LlamaIndex backend
    config = RAGConfig(backend=RAGBackend.LLAMA_INDEX)
    factory = RAGFactory(config)
    factory.build_knowledge_base("./medical_docs")
    retriever = factory.get_retriever()
    
    # Same interface for both!
    results = retriever.retrieve("chest pain guidelines", top_k=5)
"""

# =============================================================================
# Document Processing (Shared by both backends)
# =============================================================================
from .document_processor import (
    # Data classes
    DocumentType,
    DocumentMetadata,
    DocumentChunk,
    # Text processing
    TextCleaner,
    # Chunking
    ChunkingStrategy,
    DocumentChunker,
    # Main processor
    DocumentProcessor,
)

# =============================================================================
# Manual RAG Implementation
# =============================================================================
from .indexer import (
    # Stats
    IndexStats,
    # Embedding
    EmbeddingManager,
    # Indexers
    BaseIndexer,
    ChromaIndexer,
    QdrantIndexer,
    # Builder
    KnowledgeBaseBuilder,
    # Factory
    create_indexer,
)

from .retriever import (
    # Data classes
    RetrievalResult,
    RetrievalResponse,
    RetrievalStrategy,
    # Retrievers
    BaseRetriever,
    ChromaRetriever,
    QdrantRetriever,
    HybridRetriever,
    # Query processing
    MedicalQueryProcessor,
    # Factory
    create_retriever,
)

# =============================================================================
# RAG Factory (Unified Interface)
# =============================================================================
from .rag_factory import (
    RAGBackend,
    RAGConfig,
    RAGFactory,
    create_rag_system,
    compare_backends,
)

# =============================================================================
# LlamaIndex Implementation (Optional - imported on demand)
# =============================================================================
# These are imported lazily to avoid requiring LlamaIndex installation
# for users who only want to use the manual implementation.

def get_llama_index_components():
    """
    Get LlamaIndex components (lazy import)
    
    Returns dict with LlamaIndex classes, or raises ImportError if not installed.
    """
    from .llama_index_rag import (
        LlamaIndexSettings,
        LlamaIndexDocumentLoader,
        LlamaIndexIndexer,
        LlamaIndexRetriever,
        HybridMedicalRetriever,
        LlamaIndexKnowledgeBaseBuilder,
        create_llama_index_retriever,
    )
    
    return {
        "LlamaIndexSettings": LlamaIndexSettings,
        "LlamaIndexDocumentLoader": LlamaIndexDocumentLoader,
        "LlamaIndexIndexer": LlamaIndexIndexer,
        "LlamaIndexRetriever": LlamaIndexRetriever,
        "HybridMedicalRetriever": HybridMedicalRetriever,
        "LlamaIndexKnowledgeBaseBuilder": LlamaIndexKnowledgeBaseBuilder,
        "create_llama_index_retriever": create_llama_index_retriever,
    }


# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Document processor
    "DocumentType",
    "DocumentMetadata",
    "DocumentChunk",
    "TextCleaner",
    "ChunkingStrategy",
    "DocumentChunker",
    "DocumentProcessor",
    # Manual indexer
    "IndexStats",
    "EmbeddingManager",
    "BaseIndexer",
    "ChromaIndexer",
    "QdrantIndexer",
    "KnowledgeBaseBuilder",
    "create_indexer",
    # Manual retriever
    "RetrievalResult",
    "RetrievalResponse",
    "RetrievalStrategy",
    "BaseRetriever",
    "ChromaRetriever",
    "QdrantRetriever",
    "HybridRetriever",
    "MedicalQueryProcessor",
    "create_retriever",
    # RAG Factory (unified interface)
    "RAGBackend",
    "RAGConfig",
    "RAGFactory",
    "create_rag_system",
    "compare_backends",
    # LlamaIndex (lazy)
    "get_llama_index_components",
]