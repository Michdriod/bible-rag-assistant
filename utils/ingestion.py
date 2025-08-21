import asyncio
import os
from typing import List, Dict, Set
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, text
from db.db import AsyncSessionLocal, create_tables
from db.models import BibleVerse, get_verse_model, VERSION_MODELS
from utils.chunker import BibleChunker, BibleVerseChunk
from utils.embedding import get_embedding_service
import numpy as np

# Import necessary modules and libraries for Bible data ingestion
# Define the BibleIngestionService class for loading and processing data
# Implement methods to load Bible data and deduplicate chunks
class BibleIngestionService:
    def __init__(self):
        self.chunker = BibleChunker()
        self.embedding_service = get_embedding_service()
        self._processed_references: Set[str] = set()  # Track processed references
        self.supported_versions = ["kjv", "niv", "nkjv", "nlt"]
    
    async def load_bible_data(self, file_path: str) -> List[BibleVerseChunk]:
        """Load and parse Bible data from text file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Bible data file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self.chunker.chunk_bible_text(content)
        print(f"Loaded {len(chunks)} Bible verses from {file_path}")
        
        # Validate and deduplicate chunks
        chunks = self._deduplicate_chunks(chunks)
        print(f"After deduplication: {len(chunks)} unique verses")
        
        return chunks
    
    def _deduplicate_chunks(self, chunks: List[BibleVerseChunk]) -> List[BibleVerseChunk]:
        """Remove duplicate chunks based on reference"""
        seen_refs = set()
        unique_chunks = []
        
        for chunk in chunks:
            if chunk.reference not in seen_refs:
                seen_refs.add(chunk.reference)
                unique_chunks.append(chunk)
            else:
                print(f"Warning: Duplicate reference found and skipped: {chunk.reference}")
        
        return unique_chunks
    
    async def generate_embeddings(self, chunks: List[BibleVerseChunk], version: str = "kjv", batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for Bible verses with enhanced text preparation
        
        Args:
            chunks: List of Bible verse chunks
            version: Bible version (e.g., "kjv", "niv", "nkjv", "nlt")
            batch_size: Number of verses to process at once
            
        Returns:
            List of embeddings as numpy arrays
        """
        if version not in self.supported_versions:
            raise ValueError(f"Unsupported Bible version: {version}. Supported versions: {self.supported_versions}")
            
        # Prepare texts for embedding - include both verse text and reference context
        texts = []
        for chunk in chunks:
            # Combine reference and text for better semantic representation
            combined_text = f"{chunk.reference}: {chunk.text}"
            texts.append(combined_text)
        
        print(f"Generating embeddings for {len(texts)} verses ({version.upper()})...")
        
        try:
            embeddings = await self.embedding_service.encode_batch(texts, batch_size)
            print(f"Generated {len(embeddings)} embeddings for {version.upper()}")
            
            # Validate embedding dimensions
            if embeddings and len(embeddings) > 0:
                expected_dim = len(embeddings[0])
                print(f"Embedding dimension: {expected_dim}")
                
                # Check for any malformed embeddings
                for i, emb in enumerate(embeddings):
                    if len(emb) != expected_dim:
                        raise ValueError(f"Inconsistent embedding dimension at index {i}")
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings for {version}: {str(e)}")
            raise
    
    async def clear_existing_data(self, session: AsyncSession) -> int:
        """Clear existing Bible data from database (legacy method - uses default KJV table)"""
        try:
            result = await session.execute(delete(BibleVerse))
            await session.commit()
            count = result.rowcount
            print(f"Cleared {count} existing verses from default table")
            return count
        except Exception as e:
            print(f"Error clearing existing data: {str(e)}")
            await session.rollback()
            raise
            
    async def clear_existing_data_for_version(self, session: AsyncSession, version: str) -> int:
        """
        Clear existing Bible data from the specified version table
        
        Args:
            session: Database session
            version: Bible version to clear (e.g., "kjv", "niv", "nkjv", "nlt")
            
        Returns:
            Number of rows deleted
        """
        try:
            if version not in self.supported_versions:
                raise ValueError(f"Unsupported Bible version: {version}")
                
            # Get the model class for the specified version
            VerseModel = get_verse_model(version)
            
            result = await session.execute(delete(VerseModel))
            await session.commit()
            count = result.rowcount
            print(f"Cleared {count} existing verses from {version} table")
            return count
        except Exception as e:
            print(f"Error clearing existing data for {version}: {str(e)}")
            await session.rollback()
            raise
    
    async def check_existing_verses(self, session: AsyncSession) -> Dict[str, int]:
        """
        Check what verses already exist in the database (legacy method - uses default KJV table)
        """
        try:
            result = await session.execute(
                select(BibleVerse.reference, BibleVerse.id)
            )
            existing = {ref: verse_id for ref, verse_id in result.fetchall()}
            print(f"Found {len(existing)} existing verses in default table")
            return existing
        except Exception as e:
            print(f"Error checking existing verses: {str(e)}")
            return {}
            
    async def check_existing_verses_for_version(self, session: AsyncSession, version: str) -> Dict[str, int]:
        """
        Check what verses already exist in the version-specific database table
        
        Args:
            session: Database session
            version: Bible version (e.g., "kjv", "niv", "nkjv", "nlt")
            
        Returns:
            Dictionary mapping references to verse IDs
        """
        if version not in self.supported_versions:
            raise ValueError(f"Unsupported Bible version: {version}")
            
        try:
            # Get the model class for the specified version
            VerseModel = get_verse_model(version)
            
            result = await session.execute(
                select(VerseModel.reference, VerseModel.id)
            )
            existing = {ref: verse_id for ref, verse_id in result.fetchall()}
            print(f"Found {len(existing)} existing verses in {version.upper()} table")
            return existing
        except Exception as e:
            print(f"Error checking existing verses for {version}: {str(e)}")
            return {}
    
    async def insert_verses(self, session: AsyncSession, chunks: List[BibleVerseChunk], 
                          embeddings: List[np.ndarray], skip_existing: bool = True):
        """
        Insert Bible verses with embeddings into database (legacy method - uses default KJV table)
        """
        # Delegate to the version-specific method
        return await self.insert_verses_for_version(session, chunks, embeddings, "kjv", skip_existing)
    
    async def insert_verses_for_version(self, session: AsyncSession, chunks: List[BibleVerseChunk], 
                                      embeddings: List[np.ndarray], version: str = "kjv", 
                                      skip_existing: bool = True):
        """
        Insert Bible verses with embeddings into version-specific database table
        
        Args:
            session: Database session
            chunks: List of Bible verse chunks
            embeddings: List of embeddings
            version: Bible version (e.g., "kjv", "niv", "nkjv", "nlt")
            skip_existing: Whether to skip verses that already exist
        """
        if version not in self.supported_versions:
            raise ValueError(f"Unsupported Bible version: {version}")
            
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
        
        try:
            # Get the model class for the specified version
            VerseModel = get_verse_model(version)
            
            existing_verses = {}
            if skip_existing:
                existing_verses = await self.check_existing_verses_for_version(session, version)
            
            verses_to_insert = []
            skipped_count = 0
            
            for chunk, embedding in zip(chunks, embeddings):
                # Skip if verse already exists
                if skip_existing and chunk.reference in existing_verses:
                    skipped_count += 1
                    continue
                
                # Validate and prepare embedding for pgvector
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding)
                
                # Ensure embedding is the correct dimension (384 for all-MiniLM-L6-v2)
                if len(embedding) != 384:
                    raise ValueError(f"Expected embedding dimension 384, got {len(embedding)} for {chunk.reference}")
                
                # Create an instance of the appropriate version model
                verse = VerseModel(
                    book=chunk.book,
                    chapter=chunk.chapter,
                    verse=chunk.verse,
                    text=chunk.text.strip(),  # Clean up text
                    reference=chunk.reference,
                    embedding=embedding  # pgvector can handle numpy arrays directly
                )
                verses_to_insert.append(verse)
            
            if verses_to_insert:
                # Batch insert with error handling
                session.add_all(verses_to_insert)
                await session.commit()
                print(f"✅ Inserted {len(verses_to_insert)} new verses into {version.upper()} table")
            
            if skipped_count > 0:
                print(f"⏭️  Skipped {skipped_count} existing verses in {version.upper()} table")
                
        except Exception as e:
            print(f"❌ Error inserting verses for {version}: {str(e)}")
            await session.rollback()
            raise
    
    async def validate_ingestion(self, session: AsyncSession, expected_count: int) -> bool:
        """
        Validate that ingestion was successful (legacy method - uses default KJV table)
        """
        # Delegate to the version-specific method
        return await self.validate_ingestion_for_version(session, "kjv", expected_count)
    
    async def validate_ingestion_for_version(self, session: AsyncSession, version: str, expected_count: int) -> bool:
        """
        Validate that ingestion was successful for the specified version
        
        Args:
            session: Database session
            version: Bible version (e.g., "kjv", "niv", "nkjv", "nlt")
            expected_count: Expected number of verses
            
        Returns:
            True if validation passed, False otherwise
        """
        if version not in self.supported_versions:
            raise ValueError(f"Unsupported Bible version: {version}")
            
        try:
            # Get the model class for the specified version
            VerseModel = get_verse_model(version)
            
            result = await session.execute(select(VerseModel.id))
            actual_count = len(result.fetchall())
            
            print(f"Validation for {version.upper()}: Expected ≤ {expected_count}, Found {actual_count} verses in database")
            
            if actual_count == 0:
                print(f"❌ No verses found in {version.upper()} table after ingestion")
                return False
            
            # Sample a few verses to check data integrity
            sample_result = await session.execute(
                select(VerseModel.reference, VerseModel.text, VerseModel.embedding)
                .limit(3)
            )
            
            for ref, text, embedding in sample_result.fetchall():
                # Check for None or empty string values
                if not ref or not text or embedding is None:
                    print(f"❌ Invalid data found for reference: {ref} in {version.upper()} table")
                    return False
                
                # Debug: Print what we actually got
                print(f"🔍 Debug - Version: {version.upper()}, Reference: {ref}")
                print(f"🔍 Debug - Embedding type: {type(embedding)}")
                
                # Handle pgvector embedding type
                try:
                    # Convert pgvector to numpy array for validation
                    embedding_array = np.array(embedding)
                    embedding_list = embedding_array.tolist()
                    
                    print(f"🔍 Debug - Embedding shape: {embedding_array.shape}")
                    print(f"🔍 Debug - First few values: {embedding_list[:5]}")
                    
                    # Check dimension (should be 384 for all-MiniLM-L6-v2)
                    if len(embedding_array) != 384:
                        print(f"❌ Wrong embedding dimension for reference: {ref} (expected 384, got {len(embedding_array)})")
                        return False
                    
                    # Check for valid numeric values (no NaN or inf)
                    if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                        print(f"❌ Invalid values (NaN/inf) in embedding for reference: {ref}")
                        return False
                        
                    print(f"✅ Embedding validation passed for: {ref} in {version.upper()} table")
                    
                except (TypeError, ValueError, AttributeError) as e:
                    print(f"❌ Could not process embedding for reference: {ref} in {version.upper()} table - Error: {e}")
                    return False
            
            print(f"✅ Validation passed for {version.upper()}")
            return True
            
        except Exception as e:
            print(f"❌ Validation error for {version.upper()}: {str(e)}")
            return False
    
    async def ingest_bible(self, file_path: str, version: str = "kjv", clear_existing: bool = True, 
                          batch_size: int = 32, skip_existing: bool = True):
        """
        Complete ingestion process with enhanced error handling and validation
        
        Args:
            file_path: Path to the Bible text file
            version: Bible version (e.g., "kjv", "niv", "nkjv", "nlt")
            clear_existing: Whether to clear existing data for this version
            batch_size: Batch size for embedding generation
            skip_existing: Whether to skip verses that already exist
            
        Returns:
            Number of verses ingested
        """
        if version not in self.supported_versions:
            raise ValueError(f"Unsupported Bible version: {version}. Supported versions: {self.supported_versions}")
            
        try:
            print(f"🚀 Starting Bible ingestion process for version: {version.upper()}...")
            
            # Ensure tables exist
            print("📋 Ensuring database tables exist...")
            await create_tables()
            
            # Load and parse Bible data
            print(f"📖 Loading Bible data from {file_path}...")
            chunks = await self.load_bible_data(file_path)
            
            if not chunks:
                raise ValueError(f"No valid Bible verses found in the file: {file_path}")
            
            print(f"📊 Processing {len(chunks)} verses for {version.upper()}...")
            
            # Generate embeddings
            print(f"🧠 Generating embeddings for {version.upper()}...")
            embeddings = await self.generate_embeddings(chunks, version, batch_size)
            
            # Insert into database
            async with AsyncSessionLocal() as session:
                if clear_existing:
                    print(f"🧹 Clearing existing data for {version.upper()}...")
                    await self.clear_existing_data_for_version(session, version)
                
                print(f"💾 Inserting verses into {version.upper()} table...")
                await self.insert_verses_for_version(session, chunks, embeddings, version, skip_existing)
                
                # Validate ingestion
                print(f"✅ Validating ingestion for {version.upper()}...")
                validation_passed = await self.validate_ingestion_for_version(session, version, len(chunks))
                
                if not validation_passed:
                    raise ValueError(f"Ingestion validation failed for {version.upper()}")
            
            print(f"🎉 Bible ingestion completed successfully for {version.upper()}!")
            return len(chunks)
            
        except Exception as e:
            print(f"💥 Error during Bible ingestion: {str(e)}")
            raise
    

    async def get_ingestion_stats(self) -> Dict[str, any]:
        """Get statistics about all ingested Bible versions"""
        try:
            results = {}
            
            # Get stats for each version
            for version in self.supported_versions:
                stats = await self.get_ingestion_stats_for_version(version)
                results[version] = stats
                
            # Add overall totals
            total_verses = sum(stats.get("total_verses", 0) for version, stats in results.items() 
                              if isinstance(stats.get("total_verses", 0), int))
            
            results["overall"] = {
                "total_verses": total_verses,
                "versions_available": [v for v in self.supported_versions if results.get(v, {}).get("total_verses", 0) > 0]
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error getting overall ingestion stats: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {"error": str(e)}
    
    async def get_ingestion_stats_for_version(self, version: str) -> Dict[str, any]:
        """
        Get statistics about the ingested Bible data for a specific version
        
        Args:
            version: Bible version (e.g., "kjv", "niv", "nkjv", "nlt")
            
        Returns:
            Dictionary of statistics
        """
        if version not in self.supported_versions:
            return {"error": f"Unsupported version: {version}"}
        
        try:
            # Get the model class for the specified version
            VerseModel = get_verse_model(version)
            
            async with AsyncSessionLocal() as session:
                # Total verses using COUNT
                total_result = await session.execute(
                    select(func.count(VerseModel.id))
                )
                total_count = total_result.scalar()

                # Unique books count using COUNT(DISTINCT ...)
                books_result = await session.execute(
                    select(func.count(func.distinct(VerseModel.book)))
                )
                books_count = books_result.scalar()

                # Sample embedding dimension
                sample_result = await session.execute(
                    select(VerseModel.embedding).limit(1)
                )
                embedding_dim = 0
                sample_row = sample_result.fetchone()

                if sample_row and sample_row[0] is not None:
                    embedding_array = np.array(sample_row[0])
                    embedding_dim = len(embedding_array)

                return {
                    "total_verses": total_count or 0,
                    "unique_books": books_count or 0,
                    "embedding_dimension": embedding_dim
                }

        except Exception as e:
            print(f"❌ Error getting ingestion stats for {version}: {str(e)}")
            
            try:
                # Fallback query using raw SQL
                async with AsyncSessionLocal() as session:
                    from sqlalchemy import text
                    table_name = f"bible_verses_{version}"
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = result.scalar()

                    return {
                        "total_verses": count or 0,
                        "unique_books": "Error retrieving",
                        "embedding_dimension": "Error retrieving"
                    }
            except Exception as fallback_error:
                print(f"❌ Fallback stats query also failed for {version}: {str(fallback_error)}")
                return {
                    "total_verses": 0,
                    "unique_books": 0,
                    "embedding_dimension": 0,
                    "error": str(e)
                }
    #             embedding_dim = 0
    #             sample_row = sample_result.fetchone()
                
    #             if sample_row and sample_row[0] is not None:
    #                 # Handle pgvector embedding
    #                 import numpy as np
    #                 embedding_array = np.array(sample_row[0])
    #                 embedding_dim = len(embedding_array)
                
    #             return {
    #                 "total_verses": total_count or 0,
    #                 "unique_books": books_count or 0,
    #                 "embedding_dimension": embedding_dim
    #             }
                
    #         except Exception as e:
    #             print(f"❌ Error getting ingestion stats: {str(e)}")
    #             import traceback
    #             traceback.print_exc()
            
    #             # Fallback: try simpler query
    #             try:
    #                 async with AsyncSessionLocal() as session:
    #                     # Simple count query
    #                     from sqlalchemy import text
    #                     result = await session.execute(text("SELECT COUNT(*) FROM bible_verses"))
    #                     count = result.scalar()
                    
    #                     return {
    #                         "total_verses": count or 0,
    #                         "unique_books": "Error retrieving",
    #                         "embedding_dimension": "Error retrieving"
    #                     }
    #             except Exception as fallback_error:
    #                 print(f"❌ Fallback stats query also failed: {str(fallback_error)}")
    #                 return {
    #                     "total_verses": "Error retrieving",
    #                     "unique_books": "Error retrieving", 
    #                     "embedding_dimension": "Error retrieving"
    #                 }
    # async def get_ingestion_stats(self) -> Dict[str, int]:
    #     """Get statistics about the ingested Bible data"""
    #     try:
    #         async with AsyncSessionLocal() as session:
    #             # Total verses
    #             total_result = await session.execute(select(BibleVerse.id))
    #             total_count = len(total_result.fetchall())
                
    #             # Books count
    #             books_result = await session.execute(
    #                 select(BibleVerse.book).distinct()
    #             )
    #             books_count = len(books_result.fetchall())
                
    #             # Sample embedding dimension
    #             sample_result = await session.execute(
    #                 select(BibleVerse.embedding).limit(1)
    #             )
    #             embedding_dim = 0
    #             sample_embedding = sample_result.fetchone()
    #             if sample_embedding and sample_embedding[0]:
    #                 embedding_dim = len(sample_embedding[0])
                
    #             return {
    #                 "total_verses": total_count,
    #                 "unique_books": books_count,
    #                 "embedding_dimension": embedding_dim
    #             }
                
    #     except Exception as e:
    #         print(f"Error getting ingestion stats: {str(e)}")
    #         return {}

async def main():
    """Run ingestion from command line with enhanced options"""
    ingestion_service = BibleIngestionService()
    
    # Default Bible file path
    bible_file = "data/bible_cleaned.txt"
    
    if not os.path.exists(bible_file):
        print(f"❌ Bible file not found: {bible_file}")
        print("Please ensure you have a Bible text file in the correct format.")
        print("Expected format: Each line should be 'Book Chapter:Verse Text'")
        print("Example: 'John 3:16 For God so loved the world...'")
        return
    
    try:
        # Run ingestion
        count = await ingestion_service.ingest_bible(
            bible_file, 
            clear_existing=True, 
            batch_size=32,
            skip_existing=False  # Since we're clearing, no need to skip
        )
        
        # Get and display stats
        stats = await ingestion_service.get_ingestion_stats()
        print(f"\n📈 Ingestion Statistics:")
        print(f"   Total verses: {stats.get('total_verses', 'Unknown')}")
        print(f"   Unique books: {stats.get('unique_books', 'Unknown')}")
        print(f"   Embedding dimension: {stats.get('embedding_dimension', 'Unknown')}")
        
        print(f"\n🎉 Successfully ingested {count} Bible verses!")
        
    except Exception as e:
        print(f"💥 Ingestion failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())


















# import asyncio
# import os
# from typing import List, Dict, Set
# from sqlalchemy.ext.asyncio import AsyncSession
# from sqlalchemy import select, delete
# from db.db import AsyncSessionLocal, create_tables
# from db.models import BibleVerse
# from utils.chunker import BibleChunker, BibleVerseChunk
# from utils.embedding import get_embedding_service
# import numpy as np

# class BibleIngestionService:
#     def __init__(self):
#         self.chunker = BibleChunker()
#         self.embedding_service = get_embedding_service()
#         self._processed_references: Set[str] = set()  # Track processed references
    
#     async def load_bible_data(self, file_path: str) -> List[BibleVerseChunk]:
#         """Load and parse Bible data from text file"""
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"Bible data file not found: {file_path}")
        
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         chunks = self.chunker.chunk_bible_text(content)
#         print(f"Loaded {len(chunks)} Bible verses from {file_path}")
        
#         # Validate and deduplicate chunks
#         chunks = self._deduplicate_chunks(chunks)
#         print(f"After deduplication: {len(chunks)} unique verses")
        
#         return chunks
    
#     def _deduplicate_chunks(self, chunks: List[BibleVerseChunk]) -> List[BibleVerseChunk]:
#         """Remove duplicate chunks based on reference"""
#         seen_refs = set()
#         unique_chunks = []
        
#         for chunk in chunks:
#             if chunk.reference not in seen_refs:
#                 seen_refs.add(chunk.reference)
#                 unique_chunks.append(chunk)
#             else:
#                 print(f"Warning: Duplicate reference found and skipped: {chunk.reference}")
        
#         return unique_chunks
    
#     async def generate_embeddings(self, chunks: List[BibleVerseChunk], batch_size: int = 32) -> List[np.ndarray]:
#         """Generate embeddings for Bible verses with enhanced text preparation"""
#         # Prepare texts for embedding - include both verse text and reference context
#         texts = []
#         for chunk in chunks:
#             # Combine reference and text for better semantic representation
#             combined_text = f"{chunk.reference}: {chunk.text}"
#             texts.append(combined_text)
        
#         print(f"Generating embeddings for {len(texts)} verses...")
        
#         try:
#             embeddings = await self.embedding_service.encode_batch(texts, batch_size)
#             print(f"Generated {len(embeddings)} embeddings")
            
#             # Validate embedding dimensions
#             if embeddings and len(embeddings) > 0:
#                 expected_dim = len(embeddings[0])
#                 print(f"Embedding dimension: {expected_dim}")
                
#                 # Check for any malformed embeddings
#                 for i, emb in enumerate(embeddings):
#                     if len(emb) != expected_dim:
#                         raise ValueError(f"Inconsistent embedding dimension at index {i}")
            
#             return embeddings
            
#         except Exception as e:
#             print(f"Error generating embeddings: {str(e)}")
#             raise
    
#     async def clear_existing_data(self, session: AsyncSession) -> int:
#         """Clear existing Bible data from database"""
#         try:
#             result = await session.execute(delete(BibleVerse))
#             await session.commit()
#             count = result.rowcount
#             print(f"Cleared {count} existing verses")
#             return count
#         except Exception as e:
#             print(f"Error clearing existing data: {str(e)}")
#             await session.rollback()
#             raise
    
#     async def check_existing_verses(self, session: AsyncSession) -> Dict[str, int]:
#         """Check what verses already exist in the database"""
#         try:
#             result = await session.execute(
#                 select(BibleVerse.reference, BibleVerse.id)
#             )
#             existing = {ref: verse_id for ref, verse_id in result.fetchall()}
#             print(f"Found {len(existing)} existing verses in database")
#             return existing
#         except Exception as e:
#             print(f"Error checking existing verses: {str(e)}")
#             return {}
    
#     async def insert_verses(self, session: AsyncSession, chunks: List[BibleVerseChunk], 
#                           embeddings: List[np.ndarray], skip_existing: bool = True):
#         """Insert Bible verses with embeddings into database"""
#         if len(chunks) != len(embeddings):
#             raise ValueError(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
        
#         try:
#             existing_verses = {}
#             if skip_existing:
#                 existing_verses = await self.check_existing_verses(session)
            
#             verses_to_insert = []
#             skipped_count = 0
            
#             for chunk, embedding in zip(chunks, embeddings):
#                 # Skip if verse already exists
#                 if skip_existing and chunk.reference in existing_verses:
#                     skipped_count += 1
#                     continue
                
#                 # Validate embedding
#                 if not isinstance(embedding, np.ndarray):
#                     embedding = np.array(embedding)
                
#                 verse = BibleVerse(
#                     book=chunk.book,
#                     chapter=chunk.chapter,
#                     verse=chunk.verse,
#                     text=chunk.text.strip(),  # Clean up text
#                     reference=chunk.reference,
#                     embedding=embedding.tolist()  # Convert numpy array to list for pgvector
#                 )
#                 verses_to_insert.append(verse)
            
#             if verses_to_insert:
#                 # Batch insert with error handling
#                 session.add_all(verses_to_insert)
#                 await session.commit()
#                 print(f"✅ Inserted {len(verses_to_insert)} new verses into database")
            
#             if skipped_count > 0:
#                 print(f"⏭️  Skipped {skipped_count} existing verses")
                
#         except Exception as e:
#             print(f"❌ Error inserting verses: {str(e)}")
#             await session.rollback()
#             raise
    
#     async def validate_ingestion(self, session: AsyncSession, expected_count: int) -> bool:
#         """Validate that ingestion was successful"""
#         try:
#             result = await session.execute(select(BibleVerse.id))
#             actual_count = len(result.fetchall())
            
#             print(f"Validation: Expected ≤ {expected_count}, Found {actual_count} verses in database")
            
#             if actual_count == 0:
#                 print("❌ No verses found in database after ingestion")
#                 return False
            
#             # Sample a few verses to check data integrity
#             sample_result = await session.execute(
#                 select(BibleVerse.reference, BibleVerse.text, BibleVerse.embedding)
#                 .limit(3)
#             )
            
#             for ref, text, embedding in sample_result.fetchall():
#                 # Check for None or empty string values
#                 if not ref or not text or embedding is None:
#                     print(f"❌ Invalid data found for reference: {ref}")
#                     return False
                
#                 # Debug: Print what we actually got
#                 print(f"🔍 Debug - Reference: {ref}")
#                 print(f"🔍 Debug - Embedding type: {type(embedding)}")
#                 print(f"🔍 Debug - Embedding length: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
#                 print(f"🔍 Debug - First few values: {str(embedding)[:100]}...")
                
#                 # Check embedding validity - handle different possible types
#                 if isinstance(embedding, list):
#                     if len(embedding) == 0:
#                         print(f"❌ Empty embedding list for reference: {ref}")
#                         return False
#                 elif isinstance(embedding, (tuple, np.ndarray)):
#                     if len(embedding) == 0:
#                         print(f"❌ Empty embedding array for reference: {ref}")
#                         return False
#                 else:
#                     print(f"❌ Unexpected embedding type {type(embedding)} for reference: {ref}")
#                     return False
                
#                 # Additional check: ensure embedding contains numeric values
#                 try:
#                     # Convert to list if it's not already
#                     embedding_list = list(embedding) if not isinstance(embedding, list) else embedding
                    
#                     if not all(isinstance(val, (int, float)) and not np.isnan(val) for val in embedding_list[:5]):
#                         print(f"❌ Non-numeric or NaN values in embedding for reference: {ref}")
#                         return False
                        
#                     print(f"✅ Embedding validation passed for: {ref}")
                    
#                 except (TypeError, IndexError, ValueError) as e:
#                     print(f"❌ Malformed embedding for reference: {ref} - Error: {e}")
#                     return False
            
#             print("✅ Validation passed")
#             return True
            
#         except Exception as e:
#             print(f"❌ Validation error: {str(e)}")
#             return False
    
#     async def ingest_bible(self, file_path: str, clear_existing: bool = True, 
#                           batch_size: int = 32, skip_existing: bool = True):
#         """Complete ingestion process with enhanced error handling and validation"""
#         try:
#             print("🚀 Starting Bible ingestion process...")
            
#             # Ensure tables exist
#             print("📋 Ensuring database tables exist...")
#             await create_tables()
            
#             # Load and parse Bible data
#             print("📖 Loading Bible data...")
#             chunks = await self.load_bible_data(file_path)
            
#             if not chunks:
#                 raise ValueError("No valid Bible verses found in the file")
            
#             print(f"📊 Processing {len(chunks)} verses...")
            
#             # Generate embeddings
#             print("🧠 Generating embeddings...")
#             embeddings = await self.generate_embeddings(chunks, batch_size)
            
#             # Insert into database
#             async with AsyncSessionLocal() as session:
#                 if clear_existing:
#                     print("🧹 Clearing existing data...")
#                     await self.clear_existing_data(session)
                
#                 print("💾 Inserting verses into database...")
#                 await self.insert_verses(session, chunks, embeddings, skip_existing)
                
#                 # Validate ingestion
#                 print("✅ Validating ingestion...")
#                 validation_passed = await self.validate_ingestion(session, len(chunks))
                
#                 if not validation_passed:
#                     raise ValueError("Ingestion validation failed")
            
#             print("🎉 Bible ingestion completed successfully!")
#             return len(chunks)
            
#         except Exception as e:
#             print(f"💥 Error during Bible ingestion: {str(e)}")
#             raise

#     async def get_ingestion_stats(self) -> Dict[str, int]:
#         """Get statistics about the ingested Bible data"""
#         try:
#             async with AsyncSessionLocal() as session:
#                 # Total verses
#                 total_result = await session.execute(select(BibleVerse.id))
#                 total_count = len(total_result.fetchall())
                
#                 # Books count
#                 books_result = await session.execute(
#                     select(BibleVerse.book).distinct()
#                 )
#                 books_count = len(books_result.fetchall())
                
#                 # Sample embedding dimension
#                 sample_result = await session.execute(
#                     select(BibleVerse.embedding).limit(1)
#                 )
#                 embedding_dim = 0
#                 sample_embedding = sample_result.fetchone()
#                 if sample_embedding and sample_embedding[0]:
#                     embedding_dim = len(sample_embedding[0])
                
#                 return {
#                     "total_verses": total_count,
#                     "unique_books": books_count,
#                     "embedding_dimension": embedding_dim
#                 }
                
#         except Exception as e:
#             print(f"Error getting ingestion stats: {str(e)}")
#             return {}

# async def main():
#     """Run ingestion from command line"""
#     ingestion_service = BibleIngestionService()
    
#     # Default Bible file path
#     bible_file = "data/bible_cleaned.txt"
    
#     if not os.path.exists(bible_file):
#         print(f"❌ Bible file not found: {bible_file}")
#         print("Please ensure you have a Bible text file in the correct format.")
#         print("Expected format: Each line should be 'Book Chapter:Verse Text'")
#         print("Example: 'John 3:16 For God so loved the world...'")
#         return
    
#     try:
#         count = await ingestion_service.ingest_bible(bible_file)
#         print(f"🎉 Successfully ingested {count} Bible verses!")
#     except Exception as e:
#         print(f"💥 Ingestion failed: {str(e)}")

# if __name__ == "__main__":
#     asyncio.run(main())