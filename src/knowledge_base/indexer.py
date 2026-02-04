"""
Vector Store Indexer for Medical Knowledge Base

Handles indexing of document chunks into vector stores (Qdrant or ChromaDB).
Supports multiple embedding models and index management.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from .document_processor import (
    DocumentChunk,
    DocumentProcessor,
    DocumentType,
    ChunkingStrategy,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IndexStats:
    """Statistics about a vector index"""
    total_documents: int
    total_chunks: int
    embedding_dimension: int
    index_size_mb: Optional[float] = None
    
    def __str__(self) -> str:
        return (
            f"IndexStats(docs={self.total_documents}, "
            f"chunks={self.total_chunks}, dim={self.embedding_dimension})"
        )


# =============================================================================
# Embedding Manager
# =============================================================================

class EmbeddingManager:
    """
    Manage embedding model for vector generation
    
    Supports HuggingFace models and provides utilities for batch embedding.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        trust_remote_code: bool = True,
    ):
        """
        Initialize embedding manager
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu' or 'cuda')
            trust_remote_code: Whether to trust remote code
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimension = None
        self.trust_remote_code = trust_remote_code
        
        logger.info(f"Embedding manager initialized with model: {model_name}")
    
    def _load_model(self):
        """Lazy load the embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=self.trust_remote_code,
                )
                
                # Get embedding dimension
                test_embedding = self._model.encode("test")
                self._dimension = len(test_embedding)
                
                logger.info(
                    f"Loaded embedding model: {self.model_name} "
                    f"(dim={self._dimension}, device={self.device})"
                )
                
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        self._load_model()
        return self._dimension
    
    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single text string
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        self._load_model()
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Embed multiple texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
        """
        self._load_model()
        
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        
        return [emb.tolist() for emb in embeddings]
    
    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 32,
    ) -> List[List[float]]:
        """
        Embed document chunks
        
        Args:
            chunks: List of DocumentChunk objects
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)


# =============================================================================
# Abstract Base Indexer
# =============================================================================

class BaseIndexer(ABC):
    """Abstract base class for vector store indexers"""
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        collection_name: str = "medical_knowledge",
    ):
        self.embedding_manager = embedding_manager
        self.collection_name = collection_name
    
    @abstractmethod
    def create_index(self) -> bool:
        """Create a new index/collection"""
        pass
    
    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Add document chunks to the index"""
        pass
    
    @abstractmethod
    def delete_index(self) -> bool:
        """Delete the index/collection"""
        pass
    
    @abstractmethod
    def get_stats(self) -> IndexStats:
        """Get index statistics"""
        pass
    
    @abstractmethod
    def index_exists(self) -> bool:
        """Check if index exists"""
        pass


# =============================================================================
# ChromaDB Indexer
# =============================================================================

class ChromaIndexer(BaseIndexer):
    """
    ChromaDB vector store indexer
    
    Simple, lightweight vector store that works well for development
    and smaller knowledge bases.
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        collection_name: str = "medical_knowledge",
        persist_directory: Optional[str] = "./chroma_db",
    ):
        """
        Initialize ChromaDB indexer
        
        Args:
            embedding_manager: Embedding manager instance
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
        """
        super().__init__(embedding_manager, collection_name)
        
        self.persist_directory = persist_directory
        self._client = None
        self._collection = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            if self.persist_directory:
                # Persistent client
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            else:
                # In-memory client
                self._client = chromadb.Client(
                    Settings(anonymized_telemetry=False)
                )
            
            logger.info(f"ChromaDB client initialized: {self.persist_directory or 'in-memory'}")
            
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
    
    def _get_or_create_collection(self):
        """Get or create the collection"""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection
    
    def create_index(self) -> bool:
        """Create a new collection"""
        try:
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created ChromaDB collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add document chunks to the collection
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        collection = self._get_or_create_collection()
        
        # Generate embeddings
        embeddings = self.embedding_manager.embed_chunks(chunks)
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_dict() for chunk in chunks]
        
        # SANITIZATION STEP: Fix types that ChromaDB hates
        for meta in metadatas:
            # 1. Remove 'text' (it's stored as document content)
            meta.pop('text', None)
            
            # 2. Convert lists/dicts to strings
            for key, value in list(meta.items()):
                if isinstance(value, (list, tuple)):
                    # Convert ["a", "b"] -> "a, b"
                    meta[key] = ", ".join(map(str, value))
                elif isinstance(value, dict):
                    # Convert dicts to string representation
                    meta[key] = str(value)
                elif value is None:
                    # Optional: Remove None values or convert to empty string
                    meta[key] = ""
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(chunks)} chunks to ChromaDB collection")
        return len(chunks)
    
    def delete_index(self) -> bool:
        """Delete the collection"""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = None
            logger.info(f"Deleted ChromaDB collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def get_stats(self) -> IndexStats:
        """Get collection statistics"""
        collection = self._get_or_create_collection()
        count = collection.count()
        
        return IndexStats(
            total_documents=count,  # ChromaDB doesn't distinguish docs/chunks
            total_chunks=count,
            embedding_dimension=self.embedding_manager.dimension,
        )
    
    def index_exists(self) -> bool:
        """Check if collection exists"""
        try:
            collections = self._client.list_collections()
            return any(c.name == self.collection_name for c in collections)
        except:
            return False


# =============================================================================
# Qdrant Indexer
# =============================================================================

class QdrantIndexer(BaseIndexer):
    """
    Qdrant vector store indexer
    
    Production-ready vector store with advanced features like
    filtering, payload indexing, and horizontal scaling.
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        collection_name: str = "medical_knowledge",
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        """
        Initialize Qdrant indexer
        
        Args:
            embedding_manager: Embedding manager instance
            collection_name: Name of the collection
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key for Qdrant Cloud
            prefer_grpc: Whether to prefer gRPC over HTTP
        """
        super().__init__(embedding_manager, collection_name)
        
        self.host = host
        self.port = port
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self._client = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            from qdrant_client import QdrantClient
            
            if self.api_key:
                # Qdrant Cloud
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    prefer_grpc=self.prefer_grpc,
                )
            else:
                # Local Qdrant
                self._client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    prefer_grpc=self.prefer_grpc,
                )
            
            logger.info(f"Qdrant client initialized: {self.host}:{self.port}")
            
        except ImportError:
            raise ImportError(
                "qdrant-client is required. Install with: pip install qdrant-client"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def create_index(self) -> bool:
        """Create a new collection with vector configuration"""
        try:
            from qdrant_client.models import (
                VectorParams,
                Distance,
                PayloadSchemaType,
            )
            
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_manager.dimension,
                    distance=Distance.COSINE,
                ),
            )
            
            # Create payload indexes for common filter fields
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_type",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name="source_file",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            
            logger.info(f"Created Qdrant collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating Qdrant collection: {e}")
            return False
    
    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add document chunks to Qdrant
        
        Args:
            chunks: List of DocumentChunk objects
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0
        
        try:
            from qdrant_client.models import PointStruct
            
            # Ensure collection exists
            if not self.index_exists():
                self.create_index()
            
            # Generate embeddings
            embeddings = self.embedding_manager.embed_chunks(chunks)
            
            # Create points
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                points.append(PointStruct(
                    id=hash(chunk.chunk_id) % (2**63),  # Convert to int64
                    vector=embedding,
                    payload=chunk.to_dict(),
                ))
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
            
            logger.info(f"Added {len(chunks)} chunks to Qdrant collection")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error adding chunks to Qdrant: {e}")
            return 0
    
    def delete_index(self) -> bool:
        """Delete the collection"""
        try:
            self._client.delete_collection(self.collection_name)
            logger.info(f"Deleted Qdrant collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting Qdrant collection: {e}")
            return False
    
    def get_stats(self) -> IndexStats:
        """Get collection statistics"""
        try:
            info = self._client.get_collection(self.collection_name)
            
            return IndexStats(
                total_documents=info.points_count,
                total_chunks=info.points_count,
                embedding_dimension=info.config.params.vectors.size,
            )
        except Exception as e:
            logger.error(f"Error getting Qdrant stats: {e}")
            return IndexStats(0, 0, self.embedding_manager.dimension)
    
    def index_exists(self) -> bool:
        """Check if collection exists"""
        try:
            collections = self._client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except:
            return False


# =============================================================================
# Knowledge Base Builder
# =============================================================================

class KnowledgeBaseBuilder:
    """
    High-level interface for building the medical knowledge base
    
    Orchestrates document processing and indexing.
    """
    
    def __init__(
        self,
        indexer: BaseIndexer,
        processor: Optional[DocumentProcessor] = None,
    ):
        """
        Initialize knowledge base builder
        
        Args:
            indexer: Vector store indexer
            processor: Document processor (created if not provided)
        """
        self.indexer = indexer
        self.processor = processor or DocumentProcessor()
    
    def build_from_directory(
        self,
        directory: Union[str, Path],
        document_type: DocumentType = DocumentType.GENERAL,
        recursive: bool = True,
        clear_existing: bool = False,
    ) -> IndexStats:
        """
        Build knowledge base from a directory of documents
        
        Args:
            directory: Path to documents directory
            document_type: Default document type
            recursive: Process subdirectories
            clear_existing: Clear existing index first
            
        Returns:
            Index statistics
        """
        directory = Path(directory)
        
        if clear_existing and self.indexer.index_exists():
            logger.info("Clearing existing index...")
            self.indexer.delete_index()
        
        logger.info(f"Building knowledge base from: {directory}")
        
        # Collect all chunks
        all_chunks = []
        for chunk in self.processor.process_directory(
            directory,
            document_type=document_type,
            recursive=recursive,
        ):
            all_chunks.append(chunk)
        
        if not all_chunks:
            logger.warning("No chunks generated from documents")
            return IndexStats(0, 0, self.indexer.embedding_manager.dimension)
        
        logger.info(f"Generated {len(all_chunks)} chunks from documents")
        
        # Add to index
        self.indexer.add_chunks(all_chunks)
        
        stats = self.indexer.get_stats()
        logger.info(f"Knowledge base built: {stats}")
        
        return stats
    
    def add_document(
        self,
        filepath: Union[str, Path],
        document_type: DocumentType = DocumentType.GENERAL,
    ) -> int:
        """
        Add a single document to the knowledge base
        
        Args:
            filepath: Path to document
            document_type: Document type
            
        Returns:
            Number of chunks added
        """
        filepath = Path(filepath)
        
        chunks = self.processor.process_file(filepath, document_type)
        
        if chunks:
            return self.indexer.add_chunks(chunks)
        
        return 0


# =============================================================================
# Factory Function
# =============================================================================

def create_indexer(
    vector_store: str = "chroma",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    collection_name: str = "medical_knowledge",
    **kwargs,
) -> BaseIndexer:
    """
    Factory function to create an indexer
    
    Args:
        vector_store: "chroma" or "qdrant"
        embedding_model: HuggingFace embedding model name
        collection_name: Collection name
        **kwargs: Additional arguments for specific indexer
        
    Returns:
        Configured indexer instance
    """
    
    embedding_manager = EmbeddingManager(model_name=embedding_model)
    
    if vector_store.lower() == "chroma":
        return ChromaIndexer(
            embedding_manager=embedding_manager,
            collection_name=collection_name,
            persist_directory=kwargs.get("persist_directory", "./chroma_db"),
        )
    elif vector_store.lower() == "qdrant":
        return QdrantIndexer(
            embedding_manager=embedding_manager,
            collection_name=collection_name,
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 6333),
            api_key=kwargs.get("api_key"),
        )
    else:
        raise ValueError(f"Unknown vector store: {vector_store}")


def main():
    """CLI entry point for building knowledge base"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build medical knowledge base")
    parser.add_argument(
        "directory",
        type=str,
        help="Directory containing medical documents",
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default="chroma",
        choices=["chroma", "qdrant"],
        help="Vector store to use",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="medical_knowledge",
        help="Collection name",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index",
    )
    
    args = parser.parse_args()
    
    # Create indexer
    indexer = create_indexer(
        vector_store=args.vector_store,
        embedding_model=args.embedding_model,
        collection_name=args.collection,
    )
    
    # Build knowledge base
    builder = KnowledgeBaseBuilder(indexer)
    stats = builder.build_from_directory(
        args.directory,
        clear_existing=args.clear,
    )
    
    print(f"\nKnowledge base built successfully!")
    print(f"  Chunks indexed: {stats.total_chunks}")
    print(f"  Embedding dimension: {stats.embedding_dimension}")


if __name__ == "__main__":
    main()