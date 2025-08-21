from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union, Optional
import asyncio
from functools import lru_cache
import re
import logging

# Set up logging
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None  # Cache the embedding dimension
        
    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            # Cache the embedding dimension
            sample_embedding = self.model.encode(["test"])
            self.embedding_dim = len(sample_embedding[0])
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        return self.model
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        if self.embedding_dim is None:
            model = self._load_model()
            # This will set self.embedding_dim as a side effect
        return self.embedding_dim
    
    async def encode_text(self, text: Union[str, List[str]], 
                         normalize_embeddings: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text into embeddings asynchronously
        
        Args:
            text: String or list of strings to encode
            normalize_embeddings: Whether to normalize embeddings to unit vectors
        """
        loop = asyncio.get_event_loop()
        model = self._load_model()
        
        # Preprocess text if needed
        if isinstance(text, str):
            text = self._preprocess_text(text)
        elif isinstance(text, list):
            text = [self._preprocess_text(t) for t in text]
        
        # Run the encoding in a thread pool to avoid blocking
        embedding = await loop.run_in_executor(
            None, 
            lambda: model.encode(text, normalize_embeddings=normalize_embeddings)
        )
        
        return embedding
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before embedding
        
        This is particularly useful for Bible verses to ensure consistent formatting
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere with embedding quality
        # but keep basic punctuation that adds meaning
        text = re.sub(r'[^\w\s\.,;:!?\'"()-]', '', text)
        
        return text
    
    async def encode_batch(self, texts: List[str], batch_size: int = 32, 
                          normalize_embeddings: bool = True,
                          show_progress: bool = True) -> List[np.ndarray]:
        """
        Encode multiple texts in batches with better error handling and progress tracking
        
        Args:
            texts: List of texts to encode
            batch_size: Size of each batch
            normalize_embeddings: Whether to normalize embeddings
            show_progress: Whether to show progress information
        """
        if not texts:
            return []
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            current_batch = (batch_idx // batch_size) + 1
            
            if show_progress and total_batches > 1:
                print(f"Processing batch {current_batch}/{total_batches} "
                      f"({len(batch_texts)} texts)")
            
            try:
                # Encode the batch
                batch_embeddings = await self.encode_text(
                    batch_texts, 
                    normalize_embeddings=normalize_embeddings
                )
                
                # Handle both single embeddings and arrays of embeddings
                if isinstance(batch_embeddings, np.ndarray):
                    if batch_embeddings.ndim == 1:  # Single embedding
                        embeddings.append(batch_embeddings)
                    else:  # Multiple embeddings
                        embeddings.extend(batch_embeddings)
                else:
                    embeddings.extend(batch_embeddings)
                    
            except Exception as e:
                logger.error(f"Error processing batch {current_batch}: {str(e)}")
                # Log the problematic texts for debugging
                for i, text in enumerate(batch_texts):
                    logger.error(f"Batch {current_batch}, text {i}: {text[:100]}...")
                raise
        
        if show_progress:
            print(f"✅ Completed encoding {len(embeddings)} texts")
        
        return embeddings
    
    async def encode_bible_verses(self, verses: List[dict], 
                                 include_reference: bool = True,
                                 batch_size: int = 32) -> List[np.ndarray]:
        """
        Specialized method for encoding Bible verses
        
        Args:
            verses: List of verse dictionaries with 'reference' and 'text' keys
            include_reference: Whether to include the reference in the embedding
            batch_size: Batch size for processing
        """
        texts = []
        for verse in verses:
            if include_reference:
                # Combine reference and text for better semantic representation
                combined_text = f"{verse.get('reference', '')}: {verse.get('text', '')}"
            else:
                combined_text = verse.get('text', '')
            
            texts.append(combined_text)
        
        return await self.encode_batch(texts, batch_size=batch_size)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        """
        # Ensure embeddings are numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Compute cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    async def health_check(self) -> dict:
        """
        Perform a health check on the embedding service
        """
        try:
            # Test encoding a simple text
            test_text = "This is a test sentence for the embedding service."
            embedding = await self.encode_text(test_text)
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "embedding_dimension": len(embedding),
                "test_embedding_norm": float(np.linalg.norm(embedding))
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name
            }

# Global embedding service instance
@lru_cache(maxsize=1)
def get_embedding_service(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> EmbeddingService:
    """
    Get a cached embedding service instance
    
    Args:
        model_name: Name of the sentence transformer model to use
    """
    return EmbeddingService(model_name)

# Enhanced Bible book validation that integrates with your chunker
def create_book_name_normalizer():
    """
    Create a book name normalizer that's consistent with your chunker
    """
    # This should match the book_abbreviations in your chunker
    return {
        # Old Testament
        "gen": "Genesis", "genesis": "Genesis",
        "exod": "Exodus", "exodus": "Exodus", "ex": "Exodus",
        "lev": "Leviticus", "leviticus": "Leviticus",
        "num": "Numbers", "numbers": "Numbers",
        "deut": "Deuteronomy", "deuteronomy": "Deuteronomy", "dt": "Deuteronomy",
        # ... (rest of your mappings from chunker)
        "rev": "Revelation", "revelation": "Revelation", "re": "Revelation"
    }

def is_valid_bible_query(query: str, use_chunker_validation: bool = True) -> bool:
    """
    Enhanced Bible query validation that can optionally use your chunker's validation
    
    Args:
        query: The query to validate
        use_chunker_validation: Whether to use the chunker's reference format validation
    """
    if use_chunker_validation:
        # Import here to avoid circular imports
        try:
            from utils.chunker import BibleChunker
            chunker = BibleChunker()
            return chunker.is_reference_format(query.strip())
        except ImportError:
            pass  # Fall back to the original validation
    
    # Original validation logic
    query = query.strip().lower()

    # Match references like "John 3:16" or "1 Cor 13:4"
    reference_pattern = re.compile(r'\b(?:[1-3]?\s?[A-Za-z]{2,})\s+\d{1,3}:\d{1,3}\b')
    if reference_pattern.search(query):
        return True

    # Check for spiritual/religious keywords
    spiritual_keywords = [
        'god', 'jesus', 'holy', 'spirit', 'christ', 'heaven', 'lord', 
        'faith', 'pray', 'sin', 'cross', 'grace', 'mercy', 'salvation',
        'righteous', 'blessed', 'covenant', 'resurrection', 'eternal'
    ]
    
    for keyword in spiritual_keywords:
        if keyword in query:
            return True

    return False

# Import necessary modules and libraries for embedding service
# Define the EmbeddingService class for text-to-vector encoding
# Implement methods to load the model and encode text asynchronously