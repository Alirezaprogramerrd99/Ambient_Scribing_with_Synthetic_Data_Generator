"""
LlamaIndex RAG Implementation

Alternative RAG implementation using LlamaIndex framework.
Implements the same BaseRetriever interface as the manual implementation,
allowing seamless switching between approaches.

"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Import our existing interfaces and data classes
from .retriever import (
    BaseRetriever,
    RetrievalResult,
    RetrievalResponse,
    RetrievalStrategy,
    MedicalQueryProcessor,
)
from .document_processor import DocumentType, DocumentMetadata

logger = logging.getLogger(__name__)


# =============================================================================
# LlamaIndex Imports (Lazy Loading)
# =============================================================================

def _check_llama_index_installed():
    """Check if LlamaIndex is installed"""
    try:
        import llama_index.core
        return True
    except ImportError:
        return False


def _get_llama_index_modules():
    """Lazy import LlamaIndex modules"""
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        StorageContext,
        load_index_from_storage,
        Settings,
        Document,
    )
    from llama_index.core.node_parser import (
        SentenceSplitter,
        SemanticSplitterNodeParser,
    )
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.postprocessor import SimilarityPostprocessor
    
    return {
        "VectorStoreIndex": VectorStoreIndex,
        "SimpleDirectoryReader": SimpleDirectoryReader,
        "StorageContext": StorageContext,
        "load_index_from_storage": load_index_from_storage,
        "Settings": Settings,
        "Document": Document,
        "SentenceSplitter": SentenceSplitter,
        "SemanticSplitterNodeParser": SemanticSplitterNodeParser,
        "VectorIndexRetriever": VectorIndexRetriever,
        "RetrieverQueryEngine": RetrieverQueryEngine,
        "SimilarityPostprocessor": SimilarityPostprocessor,
    }


# =============================================================================
# LlamaIndex Settings Configuration
# =============================================================================

class LlamaIndexSettings:
    """
    Configure LlamaIndex global settings
    
    Centralizes configuration for embedding models, LLMs, and chunking.
    """
    
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        llm_model: Optional[str] = None,
        llm_provider: str = "ollama",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        """
        Initialize LlamaIndex settings
        
        Args:
            embedding_model: HuggingFace embedding model name
            llm_model: LLM model name (optional, for query engines)
            llm_provider: LLM provider ("ollama", "openai", "anthropic")
            chunk_size: Default chunk size
            chunk_overlap: Default chunk overlap
        """
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._configured = False
    
    def configure(self):
        """Apply settings to LlamaIndex globals"""
        if self._configured:
            return
        
        if not _check_llama_index_installed():
            raise ImportError(
                "LlamaIndex is not installed. Install with:\n"
                "pip install llama-index llama-index-embeddings-huggingface "
                "llama-index-vector-stores-chroma llama-index-vector-stores-qdrant"
            )
        
        modules = _get_llama_index_modules()
        Settings = modules["Settings"]
        
        # Configure embedding model
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=self.embedding_model,
                trust_remote_code=True,
            )
            logger.info(f"LlamaIndex embedding model: {self.embedding_model}")
        except ImportError:
            logger.warning(
                "llama-index-embeddings-huggingface not installed. "
                "Install with: pip install llama-index-embeddings-huggingface"
            )
        
        # Configure LLM if specified
        if self.llm_model:
            self._configure_llm(Settings)
        
        # Configure chunking defaults
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap
        
        self._configured = True
        logger.info("LlamaIndex settings configured")
    
    def _configure_llm(self, Settings):
        """Configure LLM based on provider"""
        
        if self.llm_provider == "ollama":
            try:
                from llama_index.llms.ollama import Ollama
                Settings.llm = Ollama(model=self.llm_model, request_timeout=300.0)
                logger.info(f"LlamaIndex LLM: Ollama/{self.llm_model}")
            except ImportError:
                logger.warning("llama-index-llms-ollama not installed")
        
        elif self.llm_provider == "openai":
            try:
                from llama_index.llms.openai import OpenAI
                Settings.llm = OpenAI(model=self.llm_model)
                logger.info(f"LlamaIndex LLM: OpenAI/{self.llm_model}")
            except ImportError:
                logger.warning("llama-index-llms-openai not installed")
        
        elif self.llm_provider == "anthropic":
            try:
                from llama_index.llms.anthropic import Anthropic
                Settings.llm = Anthropic(model=self.llm_model)
                logger.info(f"LlamaIndex LLM: Anthropic/{self.llm_model}")
            except ImportError:
                logger.warning("llama-index-llms-anthropic not installed")


# =============================================================================
# LlamaIndex Document Loader
# =============================================================================

class LlamaIndexDocumentLoader:
    """
    Document loader using LlamaIndex's readers
    
    Supports multiple file types and integrates with our DocumentMetadata.
    """
    
    def __init__(self, settings: Optional[LlamaIndexSettings] = None):
        """
        Initialize document loader
        
        Args:
            settings: LlamaIndex settings (created if not provided)
        """
        self.settings = settings or LlamaIndexSettings()
        self.settings.configure()
        
        self.modules = _get_llama_index_modules()
    
    def load_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        required_exts: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Load documents from a directory
        
        Args:
            directory: Path to directory
            recursive: Whether to search recursively
            required_exts: List of required extensions (e.g., [".txt", ".pdf"])
            
        Returns:
            List of LlamaIndex Document objects
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        SimpleDirectoryReader = self.modules["SimpleDirectoryReader"]
        
        reader = SimpleDirectoryReader(
            input_dir=str(directory),
            recursive=recursive,
            required_exts=required_exts,
            filename_as_id=True,
        )
        
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents from {directory}")
        
        # Add custom metadata
        for doc in documents:
            doc.metadata["document_type"] = self._infer_document_type(
                doc.metadata.get("file_path", "")
            )
        
        return documents
    
    def load_file(self, filepath: Union[str, Path]) -> List[Any]:
        """Load a single file"""
        filepath = Path(filepath)
        
        SimpleDirectoryReader = self.modules["SimpleDirectoryReader"]
        
        reader = SimpleDirectoryReader(input_files=[str(filepath)])
        return reader.load_data()
    
    def _infer_document_type(self, filepath: str) -> str:
        """Infer document type from filepath"""
        filepath_lower = filepath.lower()
        
        if "nice" in filepath_lower or "guideline" in filepath_lower:
            return DocumentType.NICE_GUIDELINE.value
        elif "textbook" in filepath_lower:
            return DocumentType.CLINICAL_TEXTBOOK.value
        elif "pathway" in filepath_lower:
            return DocumentType.CLINICAL_PATHWAY.value
        elif "bnf" in filepath_lower or "formulary" in filepath_lower:
            return DocumentType.DRUG_FORMULARY.value
        
        return DocumentType.GENERAL.value


# =============================================================================
# LlamaIndex Indexer
# =============================================================================

class LlamaIndexIndexer:
    """
    Build and manage LlamaIndex vector store indices
    
    Supports ChromaDB and Qdrant vector stores.
    """
    
    def __init__(
        self,
        settings: Optional[LlamaIndexSettings] = None,
        vector_store: str = "chroma",
        persist_dir: str = "./llama_index_store",
        collection_name: str = "medical_knowledge",
        **vector_store_kwargs,
    ):
        """
        Initialize indexer
        
        Args:
            settings: LlamaIndex settings
            vector_store: "chroma" or "qdrant"
            persist_dir: Directory for persistence
            collection_name: Collection/index name
            **vector_store_kwargs: Additional vector store arguments
        """
        self.settings = settings or LlamaIndexSettings()
        self.settings.configure()
        
        self.vector_store_type = vector_store
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.vector_store_kwargs = vector_store_kwargs
        
        self.modules = _get_llama_index_modules()
        self._index = None
        self._vector_store = None
    
    def _create_vector_store(self):
        """Create the vector store instance"""
        
        if self.vector_store_type == "chroma":
            return self._create_chroma_store()
        elif self.vector_store_type == "qdrant":
            return self._create_qdrant_store()
        else:
            raise ValueError(f"Unknown vector store: {self.vector_store_type}")
    
    def _create_chroma_store(self):
        """Create ChromaDB vector store"""
        try:
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore
            
            # Create ChromaDB client
            chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir / "chroma")
            )
            
            # Get or create collection
            chroma_collection = chroma_client.get_or_create_collection(
                name=self.collection_name
            )
            
            return ChromaVectorStore(chroma_collection=chroma_collection)
            
        except ImportError:
            raise ImportError(
                "ChromaDB integration not installed. Install with:\n"
                "pip install llama-index-vector-stores-chroma chromadb"
            )
    
    def _create_qdrant_store(self):
        """Create Qdrant vector store"""
        try:
            from qdrant_client import QdrantClient
            from llama_index.vector_stores.qdrant import QdrantVectorStore
            
            # Create Qdrant client
            client = QdrantClient(
                host=self.vector_store_kwargs.get("host", "localhost"),
                port=self.vector_store_kwargs.get("port", 6333),
                api_key=self.vector_store_kwargs.get("api_key"),
            )
            
            return QdrantVectorStore(
                client=client,
                collection_name=self.collection_name,
            )
            
        except ImportError:
            raise ImportError(
                "Qdrant integration not installed. Install with:\n"
                "pip install llama-index-vector-stores-qdrant qdrant-client"
            )
    
    def build_index(
        self,
        documents: List[Any],
        use_semantic_chunking: bool = False,
    ) -> Any:
        """
        Build vector index from documents
        
        Args:
            documents: List of LlamaIndex Document objects
            use_semantic_chunking: Use semantic splitter (slower but better)
            
        Returns:
            VectorStoreIndex instance
        """
        VectorStoreIndex = self.modules["VectorStoreIndex"]
        StorageContext = self.modules["StorageContext"]
        SentenceSplitter = self.modules["SentenceSplitter"]
        
        # Create vector store
        self._vector_store = self._create_vector_store()
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        
        # Choose node parser
        if use_semantic_chunking:
            try:
                SemanticSplitterNodeParser = self.modules["SemanticSplitterNodeParser"]
                from llama_index.core import Settings
                
                node_parser = SemanticSplitterNodeParser(
                    embed_model=Settings.embed_model,
                    breakpoint_percentile_threshold=95,
                )
                logger.info("Using semantic chunking")
            except Exception as e:
                logger.warning(f"Semantic chunking failed, using sentence splitter: {e}")
                node_parser = SentenceSplitter(
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
        else:
            node_parser = SentenceSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
        
        # Build index
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[node_parser],
            show_progress=True,
        )
        
        # Persist
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._index.storage_context.persist(persist_dir=str(self.persist_dir))
        
        logger.info(f"Index built and persisted to {self.persist_dir}")
        return self._index
    
    def load_index(self) -> Any:
        """Load existing index from storage"""
        StorageContext = self.modules["StorageContext"]
        load_index_from_storage = self.modules["load_index_from_storage"]
        
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"Index not found at {self.persist_dir}")
        
        # Recreate vector store connection
        self._vector_store = self._create_vector_store()
        
        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.persist_dir),
            vector_store=self._vector_store,
        )
        
        self._index = load_index_from_storage(storage_context)
        logger.info(f"Index loaded from {self.persist_dir}")
        
        return self._index
    
    def get_index(self) -> Any:
        """Get the current index (load if needed)"""
        if self._index is None:
            if self.persist_dir.exists():
                return self.load_index()
            else:
                raise ValueError("No index exists. Build one first with build_index()")
        return self._index


# =============================================================================
# LlamaIndex Retriever (Implements BaseRetriever Interface)
# =============================================================================

class LlamaIndexRetriever(BaseRetriever):
    """
    LlamaIndex-based retriever implementing the BaseRetriever interface
    
    This allows seamless switching between manual and LlamaIndex implementations.
    """
    
    def __init__(
        self,
        indexer: LlamaIndexIndexer,
        similarity_top_k: int = 5,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_top_n: int = 3,
    ):
        """
        Initialize LlamaIndex retriever
        
        Args:
            indexer: LlamaIndexIndexer instance with loaded index
            similarity_top_k: Number of results to retrieve
            use_reranker: Whether to use cross-encoder reranking
            reranker_model: Reranker model name
            reranker_top_n: Number of results after reranking
        """
        self.indexer = indexer
        self.similarity_top_k = similarity_top_k
        self.use_reranker = use_reranker
        self.reranker_model = reranker_model
        self.reranker_top_n = reranker_top_n
        
        self._retriever = None
        self._setup_retriever()
    
    def _setup_retriever(self):
        """Setup the LlamaIndex retriever"""
        index = self.indexer.get_index()
        
        # Create base retriever
        self._retriever = index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )
        
        # Add reranker if requested
        if self.use_reranker:
            try:
                from llama_index.postprocessor.cohere_rerank import CohereRerank
                # Or use sentence-transformer reranker
                from llama_index.core.postprocessor import SentenceTransformerRerank
                
                self._reranker = SentenceTransformerRerank(
                    model=self.reranker_model,
                    top_n=self.reranker_top_n,
                )
                logger.info(f"Reranker enabled: {self.reranker_model}")
            except ImportError:
                logger.warning("Reranker not available, continuing without it")
                self._reranker = None
        else:
            self._reranker = None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResponse:
        """
        Retrieve relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            RetrievalResponse with results in our standard format
        """
        import time
        start_time = time.time()
        
        # Update top_k if different from default
        if top_k != self.similarity_top_k:
            self._retriever = self.indexer.get_index().as_retriever(
                similarity_top_k=top_k
            )
        
        # Retrieve nodes
        nodes = self._retriever.retrieve(query)
        
        # Apply reranker if available
        if self._reranker and nodes:
            from llama_index.core import QueryBundle
            query_bundle = QueryBundle(query_str=query)
            nodes = self._reranker.postprocess_nodes(nodes, query_bundle)
        
        # Convert to our RetrievalResult format
        results = []
        for node in nodes:
            results.append(RetrievalResult(
                text=node.node.text,
                score=node.score if node.score else 0.0,
                chunk_id=node.node.node_id,
                metadata={
                    **node.node.metadata,
                    "start_char_idx": node.node.start_char_idx,
                    "end_char_idx": node.node.end_char_idx,
                },
            ))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=results,
            retrieval_time_ms=elapsed_ms,
            strategy="llama_index" + ("_reranked" if self._reranker else ""),
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
        from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
        
        import time
        start_time = time.time()
        
        # Build metadata filters
        filter_list = []
        for key, value in filters.items():
            filter_list.append(MetadataFilter(key=key, value=value))
        
        metadata_filters = MetadataFilters(filters=filter_list)
        
        # Create filtered retriever
        index = self.indexer.get_index()
        filtered_retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=metadata_filters,
        )
        
        # Retrieve
        nodes = filtered_retriever.retrieve(query)
        
        # Convert to our format
        results = []
        for node in nodes:
            results.append(RetrievalResult(
                text=node.node.text,
                score=node.score if node.score else 0.0,
                chunk_id=node.node.node_id,
                metadata=node.node.metadata,
            ))
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=results,
            retrieval_time_ms=elapsed_ms,
            strategy="llama_index_filtered",
        )


# =============================================================================
# Hybrid Medical Retriever (LlamaIndex + Custom Medical Processing)
# =============================================================================

class HybridMedicalRetriever(BaseRetriever):
    """
    Hybrid retriever combining LlamaIndex with custom medical processing
    
    Uses:
    - LlamaIndex for vector search
    - MedicalQueryProcessor for query expansion
    - Custom post-processing for clinical relevance
    """
    
    def __init__(
        self,
        llama_retriever: LlamaIndexRetriever,
        use_query_expansion: bool = True,
        use_clinical_filtering: bool = True,
    ):
        """
        Initialize hybrid retriever
        
        Args:
            llama_retriever: LlamaIndex retriever instance
            use_query_expansion: Use medical query expansion
            use_clinical_filtering: Filter by clinical relevance
        """
        self.llama_retriever = llama_retriever
        self.use_query_expansion = use_query_expansion
        self.use_clinical_filtering = use_clinical_filtering
        self.medical_processor = MedicalQueryProcessor()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResponse:
        """
        Hybrid retrieval with medical processing
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            RetrievalResponse with results
        """
        import time
        start_time = time.time()
        
        # Step 1: Expand query with medical knowledge
        if self.use_query_expansion:
            expanded_query = self.medical_processor.expand_query(query)
            logger.debug(f"Expanded query: {expanded_query[:100]}...")
        else:
            expanded_query = query
        
        # Step 2: Retrieve using LlamaIndex
        response = self.llama_retriever.retrieve(expanded_query, top_k=top_k * 2)
        
        # Step 3: Post-process for clinical relevance
        if self.use_clinical_filtering:
            filtered_results = self._filter_clinical_relevance(
                query, response.results
            )
        else:
            filtered_results = response.results
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            query=query,
            results=filtered_results[:top_k],
            retrieval_time_ms=elapsed_ms,
            strategy="hybrid_medical",
        )
    
    def retrieve_with_filter(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5,
        **kwargs,
    ) -> RetrievalResponse:
        """Hybrid retrieval with filters"""
        
        # Expand query
        if self.use_query_expansion:
            expanded_query = self.medical_processor.expand_query(query)
        else:
            expanded_query = query
        
        # Use LlamaIndex filtered retrieval
        response = self.llama_retriever.retrieve_with_filter(
            expanded_query, filters, top_k=top_k
        )
        
        # Update strategy
        response.strategy = "hybrid_medical_filtered"
        return response
    
    def _filter_clinical_relevance(
        self,
        original_query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Filter results by clinical relevance
        
        Simple heuristic-based filtering. In production, could use
        a clinical NER model or relevance classifier.
        """
        # Extract entities from query
        entities = self.medical_processor.extract_clinical_entities(original_query)
        
        query_symptoms = set(entities.get("symptoms", []))
        query_body_parts = set(entities.get("body_parts", []))
        
        # Score results by relevance to query entities
        scored_results = []
        for result in results:
            text_lower = result.text.lower()
            
            # Count matching entities
            symptom_matches = sum(1 for s in query_symptoms if s in text_lower)
            body_part_matches = sum(1 for b in query_body_parts if b in text_lower)
            
            # Calculate relevance boost
            relevance_boost = (symptom_matches * 0.1) + (body_part_matches * 0.05)
            
            # Adjust score
            adjusted_score = result.score + relevance_boost
            
            scored_results.append((adjusted_score, result))
        
        # Sort by adjusted score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [result for _, result in scored_results]


# =============================================================================
# Knowledge Base Builder (LlamaIndex Version)
# =============================================================================

class LlamaIndexKnowledgeBaseBuilder:
    """
    High-level builder for LlamaIndex-based knowledge base
    """
    
    def __init__(
        self,
        settings: Optional[LlamaIndexSettings] = None,
        vector_store: str = "chroma",
        persist_dir: str = "./llama_index_store",
        collection_name: str = "medical_knowledge",
    ):
        """
        Initialize builder
        
        Args:
            settings: LlamaIndex settings
            vector_store: Vector store type
            persist_dir: Persistence directory
            collection_name: Collection name
        """
        self.settings = settings or LlamaIndexSettings()
        self.loader = LlamaIndexDocumentLoader(self.settings)
        self.indexer = LlamaIndexIndexer(
            settings=self.settings,
            vector_store=vector_store,
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
    
    def build_from_directory(
        self,
        directory: Union[str, Path],
        use_semantic_chunking: bool = False,
    ) -> LlamaIndexRetriever:
        """
        Build knowledge base from directory
        
        Args:
            directory: Path to documents
            use_semantic_chunking: Use semantic chunking
            
        Returns:
            Configured LlamaIndexRetriever
        """
        # Load documents
        documents = self.loader.load_directory(directory)
        
        if not documents:
            raise ValueError(f"No documents found in {directory}")
        
        # Build index
        self.indexer.build_index(
            documents,
            use_semantic_chunking=use_semantic_chunking,
        )
        
        # Return retriever
        return LlamaIndexRetriever(self.indexer)
    
    def load_existing(self) -> LlamaIndexRetriever:
        """Load existing knowledge base"""
        self.indexer.load_index()
        return LlamaIndexRetriever(self.indexer)


# =============================================================================
# Factory Function
# =============================================================================

def create_llama_index_retriever(
    persist_dir: str = "./llama_index_store",
    vector_store: str = "chroma",
    collection_name: str = "medical_knowledge",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    use_hybrid: bool = True,
    use_reranker: bool = False,
    **kwargs,
) -> BaseRetriever:
    """
    Factory function to create LlamaIndex-based retriever
    
    Args:
        persist_dir: Index persistence directory
        vector_store: "chroma" or "qdrant"
        collection_name: Collection name
        embedding_model: Embedding model name
        use_hybrid: Use hybrid medical retriever
        use_reranker: Use cross-encoder reranking
        
    Returns:
        Configured retriever instance
    """
    # Create settings
    settings = LlamaIndexSettings(
        embedding_model=embedding_model,
        chunk_size=kwargs.get("chunk_size", 512),
        chunk_overlap=kwargs.get("chunk_overlap", 50),
    )
    
    # Create indexer
    indexer = LlamaIndexIndexer(
        settings=settings,
        vector_store=vector_store,
        persist_dir=persist_dir,
        collection_name=collection_name,
        **kwargs,
    )
    
    # Load or raise error
    try:
        indexer.load_index()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No index found at {persist_dir}. "
            "Build one first using LlamaIndexKnowledgeBaseBuilder"
        )
    
    # Create base retriever
    base_retriever = LlamaIndexRetriever(
        indexer=indexer,
        use_reranker=use_reranker,
    )
    
    # Wrap with hybrid if requested
    if use_hybrid:
        return HybridMedicalRetriever(
            llama_retriever=base_retriever,
            use_query_expansion=True,
            use_clinical_filtering=True,
        )
    
    return base_retriever


if __name__ == "__main__":
    print("LlamaIndex RAG Module")
    print("=" * 60)
    print()
    print("This module provides LlamaIndex-based RAG implementation")
    print("that implements the same BaseRetriever interface as the manual")
    print("implementation, allowing seamless switching between approaches.")
    print()
    print("Key Classes:")
    print("  - LlamaIndexSettings: Configure LlamaIndex globals")
    print("  - LlamaIndexDocumentLoader: Load documents")
    print("  - LlamaIndexIndexer: Build and manage indices")
    print("  - LlamaIndexRetriever: Retrieve with LlamaIndex")
    print("  - HybridMedicalRetriever: LlamaIndex + Medical processing")
    print()
    print("Usage:")
    print("  from src.knowledge_base import create_llama_index_retriever")
    print("  retriever = create_llama_index_retriever('./index')")
    print("  results = retriever.retrieve('chest pain guidelines')")