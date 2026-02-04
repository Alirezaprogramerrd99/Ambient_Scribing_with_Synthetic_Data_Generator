"""
Document Processor for Medical Knowledge Base

Handles parsing, chunking, and preprocessing of medical documents
including NICE guidelines, clinical texts, and other medical sources.

Author: Alireza Rashidi
MSc Project: Trustworthy SLMs for Ambient Clinical Scribing
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class DocumentType(str, Enum):
    """Types of medical documents"""
    NICE_GUIDELINE = "nice_guideline"
    CLINICAL_TEXTBOOK = "clinical_textbook"
    RESEARCH_PAPER = "research_paper"
    CLINICAL_PATHWAY = "clinical_pathway"
    DRUG_FORMULARY = "drug_formulary"
    PATIENT_INFORMATION = "patient_information"
    GENERAL = "general"


@dataclass
class DocumentMetadata:
    """Metadata for a medical document"""
    
    source_file: str
    document_type: DocumentType = DocumentType.GENERAL
    title: Optional[str] = None
    version: Optional[str] = None
    publication_date: Optional[str] = None
    author: Optional[str] = None
    specialty: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    url: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "source_file": self.source_file,
            "document_type": self.document_type.value,
            "title": self.title,
            "version": self.version,
            "publication_date": self.publication_date,
            "author": self.author,
            "specialty": self.specialty,
            "keywords": self.keywords,
            "url": self.url,
        }


@dataclass
class DocumentChunk:
    """A chunk of text from a document with metadata"""
    
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    chunk_index: int
    total_chunks: int
    section_title: Optional[str] = None
    start_char: int = 0
    end_char: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "section_title": self.section_title,
            "start_char": self.start_char,
            "end_char": self.end_char,
            **self.metadata.to_dict(),
        }
    
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())


# =============================================================================
# Text Cleaning
# =============================================================================

class TextCleaner:
    """Clean and normalize medical text"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove special characters but keep medical symbols
        # Keep: α, β, γ, μ, ±, °, ≤, ≥, etc.
        text = re.sub(r'[^\w\s\.,;:!?\-\'\"()\[\]{}/@#$%&*+=<>°±≤≥αβγδμ\n]', '', text)
        
        # Fix common OCR errors in medical text
        text = TextCleaner._fix_medical_ocr_errors(text)
        
        # Normalize medical abbreviations
        text = TextCleaner._normalize_abbreviations(text)
        
        return text.strip()
    
    @staticmethod
    def _fix_medical_ocr_errors(text: str) -> str:
        """Fix common OCR errors in medical documents"""
        
        corrections = {
            r'\bl\s*\.v\.': 'i.v.',  # intravenous
            r'\bm\s*g\b': 'mg',      # milligrams
            r'\bm\s*l\b': 'ml',      # milliliters
            r'\bm\s*cg\b': 'mcg',    # micrograms
            r'0(?=mg|ml|mcg)': 'O',  # Zero vs O before units
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def _normalize_abbreviations(text: str) -> str:
        """Normalize common medical abbreviations"""
        
        # Map of abbreviations to normalized forms
        abbreviations = {
            r'\bpt\b': 'patient',
            r'\bpts\b': 'patients',
            r'\bhx\b': 'history',
            r'\bdx\b': 'diagnosis',
            r'\btx\b': 'treatment',
            r'\brx\b': 'prescription',
            r'\bsx\b': 'symptoms',
            r'\bf/u\b': 'follow-up',
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without',
            r'\bb/l\b': 'bilateral',
            r'\bc/o\b': 'complains of',
        }
        
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    @staticmethod
    def extract_sections(text: str) -> List[Tuple[str, str]]:
        """
        Extract sections from a document based on headers
        
        Args:
            text: Document text
            
        Returns:
            List of (section_title, section_content) tuples
        """
        
        # Common medical document section patterns
        section_patterns = [
            r'^#+\s*(.+?)$',                    # Markdown headers
            r'^([A-Z][A-Z\s]+):?\s*$',          # ALL CAPS headers
            r'^\d+\.\s*([A-Z].+?)$',            # Numbered sections
            r'^([A-Z][a-z]+(?:\s+[A-Za-z]+)*):$',  # Title Case with colon
        ]
        
        combined_pattern = '|'.join(f'({p})' for p in section_patterns)
        
        sections = []
        current_title = "Introduction"
        current_content = []
        
        for line in text.split('\n'):
            is_header = False
            
            for pattern in section_patterns:
                match = re.match(pattern, line.strip(), re.MULTILINE)
                if match:
                    # Save previous section
                    if current_content:
                        sections.append((
                            current_title,
                            '\n'.join(current_content).strip()
                        ))
                    
                    # Start new section
                    current_title = match.group(1).strip()
                    current_content = []
                    is_header = True
                    break
            
            if not is_header:
                current_content.append(line)
        
        # Don't forget the last section
        if current_content:
            sections.append((
                current_title,
                '\n'.join(current_content).strip()
            ))
        
        return sections


# =============================================================================
# Chunking Strategies
# =============================================================================

class ChunkingStrategy(str, Enum):
    """Available chunking strategies"""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    SECTION = "section"


class DocumentChunker:
    """
    Chunk documents using various strategies
    
    Supports multiple chunking approaches optimized for medical text.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    ):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target chunk size in characters (for fixed_size) or tokens (approximate)
            chunk_overlap: Overlap between chunks
            strategy: Chunking strategy to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.cleaner = TextCleaner()
    
    def chunk_document(
        self,
        text: str,
        metadata: DocumentMetadata,
    ) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces
        
        Args:
            text: Document text
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        
        # Clean text first
        text = self.cleaner.clean_text(text)
        
        if not text:
            return []
        
        # Choose chunking method
        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            chunks = self._chunk_by_sentence(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            chunks = self._chunk_by_paragraph(text)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(text)
        elif self.strategy == ChunkingStrategy.SECTION:
            chunks = self._chunk_by_section(text)
        else:
            chunks = self._chunk_fixed_size(text)
        
        # Create DocumentChunk objects
        doc_chunks = []
        for i, (chunk_text, section_title, start, end) in enumerate(chunks):
            chunk_id = self._generate_chunk_id(metadata.source_file, i, chunk_text)
            
            doc_chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                chunk_index=i,
                total_chunks=len(chunks),
                section_title=section_title,
                start_char=start,
                end_char=end,
            ))
        
        logger.debug(f"Created {len(doc_chunks)} chunks from {metadata.source_file}")
        return doc_chunks
    
    def _generate_chunk_id(self, source: str, index: int, text: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        source_hash = hashlib.md5(source.encode()).hexdigest()[:8]
        return f"{source_hash}_{index:04d}_{content_hash}"
    
    def _chunk_fixed_size(self, text: str) -> List[Tuple[str, Optional[str], int, int]]:
        """Simple fixed-size chunking with overlap"""
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = end - int(self.chunk_size * 0.2)
                sentence_end = self._find_sentence_boundary(text, search_start, end)
                if sentence_end > search_start:
                    end = sentence_end
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append((chunk_text, None, start, end))
            
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            
            # Prevent infinite loop
            if end >= len(text):
                break
        
        return chunks
    
    def _chunk_by_sentence(self, text: str) -> List[Tuple[str, Optional[str], int, int]]:
        """Chunk by sentences, grouping to reach target size"""
        
        # Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_end = chunk_start + len(chunk_text)
                chunks.append((chunk_text, None, chunk_start, chunk_end))
                
                # Start new chunk with overlap
                overlap_sentences = []
                overlap_length = 0
                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
                chunk_start = chunk_end - overlap_length
            
            current_chunk.append(sentence)
            current_length += len(sentence)
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, None, chunk_start, chunk_start + len(chunk_text)))
        
        return chunks
    
    def _chunk_by_paragraph(self, text: str) -> List[Tuple[str, Optional[str], int, int]]:
        """Chunk by paragraphs"""
        
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                pos += 2  # Account for \n\n
                continue
            
            if current_length + len(para) > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append((chunk_text, None, chunk_start, chunk_start + len(chunk_text)))
                current_chunk = []
                current_length = 0
                chunk_start = pos
            
            current_chunk.append(para)
            current_length += len(para)
            pos += len(para) + 2
        
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append((chunk_text, None, chunk_start, chunk_start + len(chunk_text)))
        
        return chunks
    
    def _chunk_semantic(self, text: str) -> List[Tuple[str, Optional[str], int, int]]:
        """
        Semantic chunking - tries to keep related content together
        
        Uses section headers and paragraph boundaries intelligently.
        """
        
        # First extract sections
        sections = self.cleaner.extract_sections(text)
        
        if not sections:
            # Fall back to paragraph chunking
            return self._chunk_by_paragraph(text)
        
        chunks = []
        pos = 0
        
        for section_title, section_content in sections:
            if not section_content:
                continue
            
            # If section is small enough, keep it whole
            if len(section_content) <= self.chunk_size:
                chunks.append((section_content, section_title, pos, pos + len(section_content)))
                pos += len(section_content) + len(section_title) + 2
            else:
                # Chunk the section content
                section_chunks = self._chunk_by_paragraph(section_content)
                for chunk_text, _, start, end in section_chunks:
                    chunks.append((chunk_text, section_title, pos + start, pos + end))
                pos += len(section_content) + len(section_title) + 2
        
        return chunks
    
    def _chunk_by_section(self, text: str) -> List[Tuple[str, Optional[str], int, int]]:
        """Chunk by document sections only"""
        
        sections = self.cleaner.extract_sections(text)
        
        chunks = []
        pos = 0
        
        for section_title, section_content in sections:
            if section_content:
                chunks.append((
                    section_content,
                    section_title,
                    pos,
                    pos + len(section_content)
                ))
            pos += len(section_content) + len(section_title) + 2
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within a range"""
        
        search_text = text[start:end]
        
        # Look for sentence endings
        for pattern in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            last_pos = search_text.rfind(pattern)
            if last_pos != -1:
                return start + last_pos + len(pattern)
        
        # Fall back to other boundaries
        for pattern in ['; ', ': ', '\n']:
            last_pos = search_text.rfind(pattern)
            if last_pos != -1:
                return start + last_pos + len(pattern)
        
        return end


# =============================================================================
# Document Processor
# =============================================================================

class DocumentProcessor:
    """
    Main document processor for the knowledge base
    
    Handles loading, parsing, and chunking of medical documents.
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.html'}
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC,
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking
        """
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy,
        )
        self.cleaner = TextCleaner()
    
    def process_file(
        self,
        filepath: Path,
        document_type: DocumentType = DocumentType.GENERAL,
        additional_metadata: Optional[Dict] = None,
    ) -> List[DocumentChunk]:
        """
        Process a single file
        
        Args:
            filepath: Path to file
            document_type: Type of document
            additional_metadata: Additional metadata to include
            
        Returns:
            List of document chunks
        """
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if filepath.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            logger.warning(f"Unsupported file type: {filepath.suffix}")
            return []
        
        # Extract text based on file type
        text = self._extract_text(filepath)
        
        if not text:
            logger.warning(f"No text extracted from: {filepath}")
            return []
        
        # Create metadata
        metadata = DocumentMetadata(
            source_file=str(filepath),
            document_type=document_type,
            title=filepath.stem,
        )
        
        # Add additional metadata
        if additional_metadata:
            for key, value in additional_metadata.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
        
        # Chunk the document
        chunks = self.chunker.chunk_document(text, metadata)
        
        logger.info(f"Processed {filepath.name}: {len(chunks)} chunks")
        return chunks
    
    def process_directory(
        self,
        directory: Path,
        document_type: DocumentType = DocumentType.GENERAL,
        recursive: bool = True,
    ) -> Generator[DocumentChunk, None, None]:
        """
        Process all documents in a directory
        
        Args:
            directory: Directory path
            document_type: Default document type
            recursive: Whether to process subdirectories
            
        Yields:
            DocumentChunk objects
        """
        
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Get all files
        if recursive:
            files = list(directory.rglob("*"))
        else:
            files = list(directory.glob("*"))
        
        # Filter to supported files
        files = [
            f for f in files 
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        
        logger.info(f"Processing {len(files)} files from {directory}")
        
        for filepath in files:
            try:
                # Determine document type from path
                doc_type = self._infer_document_type(filepath, document_type)
                
                chunks = self.process_file(filepath, doc_type)
                for chunk in chunks:
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Error processing {filepath}: {e}")
                continue
    
    def _extract_text(self, filepath: Path) -> str:
        """Extract text from a file based on its type"""
        
        suffix = filepath.suffix.lower()
        
        if suffix == '.txt':
            return self._read_text_file(filepath)
        elif suffix == '.md':
            return self._read_text_file(filepath)
        elif suffix == '.pdf':
            return self._extract_pdf_text(filepath)
        elif suffix == '.docx':
            return self._extract_docx_text(filepath)
        elif suffix == '.html':
            return self._extract_html_text(filepath)
        else:
            return ""
    
    def _read_text_file(self, filepath: Path) -> str:
        """Read a plain text file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encodings
            for encoding in ['latin-1', 'cp1252']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        return f.read()
                except:
                    continue
        return ""
    
    def _extract_pdf_text(self, filepath: Path) -> str:
        """Extract text from PDF"""
        try:
            import pypdf
            
            text_parts = []
            with open(filepath, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            
            return '\n\n'.join(text_parts)
            
        except ImportError:
            logger.warning("pypdf not installed. Cannot process PDF files.")
            return ""
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def _extract_docx_text(self, filepath: Path) -> str:
        """Extract text from DOCX"""
        try:
            from docx import Document
            
            doc = Document(filepath)
            text_parts = []
            
            for para in doc.paragraphs:
                text_parts.append(para.text)
            
            return '\n\n'.join(text_parts)
            
        except ImportError:
            logger.warning("python-docx not installed. Cannot process DOCX files.")
            return ""
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def _extract_html_text(self, filepath: Path) -> str:
        """Extract text from HTML"""
        try:
            from bs4 import BeautifulSoup
            
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            return soup.get_text(separator='\n')
            
        except ImportError:
            logger.warning("beautifulsoup4 not installed. Cannot process HTML files.")
            return ""
        except Exception as e:
            logger.error(f"Error extracting HTML text: {e}")
            return ""
    
    def _infer_document_type(
        self,
        filepath: Path,
        default: DocumentType
    ) -> DocumentType:
        """Infer document type from filepath"""
        
        path_lower = str(filepath).lower()
        
        if 'nice' in path_lower or 'guideline' in path_lower:
            return DocumentType.NICE_GUIDELINE
        elif 'textbook' in path_lower:
            return DocumentType.CLINICAL_TEXTBOOK
        elif 'pathway' in path_lower:
            return DocumentType.CLINICAL_PATHWAY
        elif 'bnf' in path_lower or 'formulary' in path_lower:
            return DocumentType.DRUG_FORMULARY
        elif 'patient' in path_lower and 'info' in path_lower:
            return DocumentType.PATIENT_INFORMATION
        
        return default


if __name__ == "__main__":
    # Test document processor
    print("Testing Document Processor")
    print("=" * 60)
    
    # Test text cleaning
    cleaner = TextCleaner()
    sample_text = """
    CLINICAL GUIDELINE: Chest Pain Assessment
    
    Overview:
    The pt presents with hx of chest pain. Dx considerations include:
    - Acute coronary syndrome
    - Musculoskeletal pain
    
    Treatment:
    Consider tx with aspirin 300mg stat.
    F/u in 2 weeks.
    """
    
    cleaned = cleaner.clean_text(sample_text)
    print("Cleaned text sample:")
    print(cleaned[:200])
    print()
    
    # Test section extraction
    sections = cleaner.extract_sections(sample_text)
    print(f"Extracted {len(sections)} sections:")
    for title, content in sections:
        print(f"  - {title}: {len(content)} chars")
    
    # Test chunking
    chunker = DocumentChunker(
        chunk_size=200,
        chunk_overlap=20,
        strategy=ChunkingStrategy.SEMANTIC
    )
    
    metadata = DocumentMetadata(
        source_file="test.txt",
        document_type=DocumentType.NICE_GUIDELINE,
        title="Chest Pain Guidelines"
    )
    
    chunks = chunker.chunk_document(sample_text, metadata)
    print(f"\nCreated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - Chunk {chunk.chunk_index}: {chunk.word_count} words, section: {chunk.section_title}")
    
    print("\n✓ Document processor tests passed!")