# Import necessary modules and libraries for database models
from sqlalchemy import Column, Integer, String, Text, Float, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
import numpy as np

# Define base class for Bible verses with common functionality
Base = declarative_base()

class BibleVerseBase:
    """Base class with common methods for Bible verse models"""
    
    @property
    def embedding_as_list(self):
        """Convert pgvector embedding to Python list"""
        if self.embedding is None:
            return None
        # pgvector objects can be converted to numpy arrays, then to lists
        if hasattr(self.embedding, '__array__'):
            return np.array(self.embedding).tolist()
        elif hasattr(self.embedding, 'tolist'):
            return self.embedding.tolist()
        else:
            # Fallback: try to convert directly
            return list(self.embedding)
    
    @property 
    def embedding_as_numpy(self):
        """Convert pgvector embedding to numpy array"""
        if self.embedding is None:
            return None
        return np.array(self.embedding)

# Define models for different Bible versions with specific attributes

# KJV Version
class KJVVerse(Base, BibleVerseBase):
    __tablename__ = "kjv"
    
    book_id = Column(Integer, primary_key=True, index=True)
    book = Column(String(50), nullable=False, index=True)
    chapter = Column(Integer, nullable=False, index=True)
    verse = Column(Integer, nullable=False, index=True)
    text = Column(Text, nullable=False)
    reference = Column(String(100), nullable=False, index=True)  # "John 3:16"
    embedding = Column(Vector(384))  # sentence-transformers/all-MiniLM-L6-v2 produces 384-dim vectors
    
    def __repr__(self):
        return f"<KJVVerse(reference='{self.reference}', text='{self.text[:50]}...')>"

# NIV Version
class NIVVerse(Base, BibleVerseBase):
    __tablename__ = "niv"
    
    book_id = Column(Integer, primary_key=True, index=True)
    book = Column(String(50), nullable=False, index=True)
    chapter = Column(Integer, nullable=False, index=True)
    verse = Column(Integer, nullable=False, index=True)
    text = Column(Text, nullable=False)
    reference = Column(String(100), nullable=False, index=True)
    embedding = Column(Vector(384))
    
    def __repr__(self):
        return f"<NIVVerse(reference='{self.reference}', text='{self.text[:50]}...')>"

# NKJV Version
class NKJVVerse(Base, BibleVerseBase):
    __tablename__ = "nkjv"
    
    book_id = Column(Integer, primary_key=True, index=True)
    book = Column(String(50), nullable=False, index=True)
    chapter = Column(Integer, nullable=False, index=True)
    verse = Column(Integer, nullable=False, index=True)
    text = Column(Text, nullable=False)
    reference = Column(String(100), nullable=False, index=True)
    embedding = Column(Vector(384))
    
    def __repr__(self):
        return f"<NKJVVerse(reference='{self.reference}', text='{self.text[:50]}...')>"

# NLT Version
class NLTVerse(Base, BibleVerseBase):
    __tablename__ = "nlt"
    
    book_id = Column(Integer, primary_key=True, index=True)
    book = Column(String(50), nullable=False, index=True)
    chapter = Column(Integer, nullable=False, index=True)
    verse = Column(Integer, nullable=False, index=True)
    text = Column(Text, nullable=False)
    reference = Column(String(100), nullable=False, index=True)
    embedding = Column(Vector(384))
    
    def __repr__(self):
        return f"<NLTVerse(reference='{self.reference}', text='{self.text[:50]}...')>"

# For backward compatibility with existing code
BibleVerse = KJVVerse

# Dictionary to map version name strings to model classes
VERSION_MODELS = {
    "kjv": KJVVerse,
    "niv": NIVVerse,
    "nkjv": NKJVVerse,
    "nlt": NLTVerse,
}

# Get model class by version name
def get_verse_model(version):
    """Get the appropriate SQLAlchemy model class for a given Bible version"""
    if version not in VERSION_MODELS:
        raise ValueError(f"Unknown Bible version: {version}")
    return VERSION_MODELS[version]







