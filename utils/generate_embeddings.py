#!/usr/bin/env python
# Create a dedicated script for generating embeddings for existing Bible verses
# This is separate from ingestion since the data is already in the database

import asyncio
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from sqlalchemy import text
from db.db import AsyncSessionLocal, create_tables
from db.models import get_verse_model, VERSION_MODELS
from utils.embedding import get_embedding_service
import numpy as np
from pgvector.sqlalchemy import Vector as PgVector
from pgvector.asyncpg import register_vector
from pgvector import Vector
import ast


# Load environment variables
load_dotenv()

# List of supported Bible versions
SUPPORTED_VERSIONS = ["kjv", "niv", "nkjv", "nlt"]

class EmbeddingGenerator:
    """
    Generate embeddings for Bible verses already in the database
    """
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.supported_versions = SUPPORTED_VERSIONS
    
    async def generate_embeddings_for_version(self, version: str, batch_size: int = 32, 
                                              force_regenerate: bool = False):
        """
        Generate embeddings for all verses in a specific Bible version
        
        Args:
            version: Bible version to process (e.g., "kjv", "niv", "nkjv", "nlt")
            batch_size: Number of verses to process in each batch
            force_regenerate: If True, regenerate embeddings even if they already exist
        """
        if version not in self.supported_versions:
            raise ValueError(f"Unsupported Bible version: {version}. Supported versions: {self.supported_versions}")
            
        print(f"🚀 Generating embeddings for {version.upper()}...")
        
        # Get the model class for the specified version
        VerseModel = get_verse_model(version)
        
        async with AsyncSessionLocal() as session:
            # Count total verses
            where_clause = "" if force_regenerate else "WHERE embedding IS NULL"
            count_query = f"SELECT COUNT(*) FROM {version} {where_clause}"
            result = await session.execute(text(count_query))
            total_count = result.scalar()
            
            if total_count == 0:
                print(f"✅ No verses need embedding generation in {version.upper()}.")
                return 0
                
            print(f"📊 Found {total_count} verses in {version.upper()} that need embeddings.")
            
            # Process in batches to avoid memory issues
            processed_count = 0

            # Use repeated WHERE ... IS NULL LIMIT pagination (no OFFSET) to avoid skipping
            batch_num = 0
            while True:
                # Fetch next batch of verses that still need embeddings
                where_clause = "" if force_regenerate else "WHERE embedding IS NULL"
                fetch_query = f"""
                    SELECT book_id, book, chapter, verse, text, reference 
                    FROM {version} {where_clause}
                    ORDER BY book, chapter, verse
                    LIMIT {batch_size}
                """
                verses_result = await session.execute(text(fetch_query))
                batch_verses = verses_result.fetchall()

                if not batch_verses:
                    break

                batch_num += 1
                print(f"⚙️ Processing batch {batch_num} for {version.upper()}...")
                
                # Prepare texts for embedding
                texts = []
                for verse in batch_verses:
                    # For semantic search, include reference in the embedding
                    combined_text = f"{verse.reference}: {verse.text}"
                    texts.append(combined_text)
                
                # Generate embeddings
                embeddings = await self.embedding_service.encode_batch(
                    texts, batch_size=batch_size, show_progress=True
                )
                
                # Update database≈
                for verse, embedding in zip(batch_verses, embeddings):
                    # Format embedding for pgvector - it needs a specific string format
                    # Convert numpy array to a string representation for pgvector: '[1,2,3]'
                    
                    # Convert embedding list to PostgreSQL array string
                    embedding_list = embedding.tolist()
                    embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
                    update_query = f"""
                            UPDATE {version}
                            SET embedding = :embedding
                            WHERE book_id = :book_id AND chapter = :chapter AND verse = :verse
                        """
                    await session.execute(text(update_query), {
                            "embedding": embedding_str,
                            "book_id": verse.book_id,
                            "chapter": verse.chapter,
                            "verse": verse.verse
                        })
                
                # Commit after each batch
                await session.commit()

                # Update counters
                processed_count += len(batch_verses)
                print(f"✅ Updated {processed_count}/{total_count} embeddings in {version.upper()}.")
            
        print(f"🎉 Completed embedding generation for {version.upper()}. Updated {processed_count} verses.")
        return processed_count
    
    async def verify_embeddings(self, version: str, sample_size: int = 5):
        """
        Verify that embeddings were generated correctly for a Bible version
        
        Args:
            version: Bible version to check (e.g., "kjv", "niv", "nkjv", "nlt")
            sample_size: Number of random samples to check
        """
        if version not in self.supported_versions:
            raise ValueError(f"Unsupported Bible version: {version}. Supported versions: {self.supported_versions}")
            
        print(f"🔍 Verifying embeddings for {version.upper()}...")
        
        async with AsyncSessionLocal() as session:
            # Check total embeddings
            count_query = f"SELECT COUNT(*) FROM {version} WHERE embedding IS NOT NULL"
            result = await session.execute(text(count_query))
            with_embeddings = result.scalar()
            
            # Check total verses
            total_query = f"SELECT COUNT(*) FROM {version}"
            result = await session.execute(text(total_query))
            total_verses = result.scalar()
            
            # Calculate percentage
            if total_verses > 0:
                percentage = (with_embeddings / total_verses) * 100
                print(f"📊 {with_embeddings}/{total_verses} verses have embeddings ({percentage:.2f}%).")
            else:
                print(f"⚠️ No verses found in {version.upper()}.")
                return False
            
            if with_embeddings == 0:
                print(f"❌ No embeddings found in {version.upper()}.")
                return False
            
            # Check random samples
            sample_query = f"""
                SELECT book, chapter, verse, reference, embedding 
                FROM {version} 
                WHERE embedding IS NOT NULL
                ORDER BY RANDOM() 
                LIMIT {sample_size}
            """
            
            samples = await session.execute(text(sample_query))
            all_valid = True
            
            for sample in samples:
                # Check embedding
                if not sample.embedding:
                    print(f"❌ Missing embedding for {sample.reference}")
                    all_valid = False
                    continue
                # parse embedding string to list if needed
                embedding_val = sample.embedding
                if isinstance(embedding_val, str):
                    try:
                        embedding_list = ast.literal_eval(embedding_val)
                    except Exception:
                        print(f"❌ Could not parse embedding for {sample.reference}")
                        all_valid = False
                        continue
                else:
                    embedding_list = embedding_val
                    
                embedding_array = np.array(embedding_list)
                if len(embedding_array) != 384:
                    print(f"❌ Wrong embedding dimension for {sample.reference}: {len(embedding_array)} (expected 384)")
                    all_valid = False
                    continue
                
                
                
                
                # # Check embedding dimension
                # embedding_array = np.array(sample.embedding)
                # if len(embedding_array) != 384:  # Expected dimension for all-MiniLM-L6-v2
                #     print(f"❌ Wrong embedding dimension for {sample.reference}: {len(embedding_array)} (expected 384)")
                #     all_valid = False
                #     continue
                
                
                
                
                # Check for NaN or Inf
                if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                    print(f"❌ Invalid values in embedding for {sample.reference}")
                    all_valid = False
                    continue
                
                print(f"✅ Valid embedding for {sample.reference}: dim={len(embedding_array)}")
            
            return all_valid
    
    async def generate_all_embeddings(self, force_regenerate: bool = False):
        """
        Generate embeddings for all supported Bible versions
        
        Args:
            force_regenerate: If True, regenerate all embeddings even if they already exist
        """
        print("🚀 Starting embedding generation for all Bible versions...")
        total_processed = 0
        
        for version in self.supported_versions:
            try:
                processed = await self.generate_embeddings_for_version(
                    version, force_regenerate=force_regenerate
                )
                total_processed += processed
                
                # Verify embeddings
                verified = await self.verify_embeddings(version)
                status = "✅ Verified" if verified else "⚠️ Verification failed"
                print(f"{status} for {version.upper()}")
                
                print("-" * 60)
            except Exception as e:
                print(f"❌ Error processing {version.upper()}: {str(e)}")
                import traceback
                traceback.print_exc()
                print("-" * 60)
        
        print(f"🎉 Completed embedding generation. Total verses processed: {total_processed}")

async def main():
    """Main entry point for embedding generation"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for Bible verses")
    parser.add_argument("--version", "-v", type=str, choices=SUPPORTED_VERSIONS,
                        help="Specific Bible version to process (e.g., kjv, niv, nkjv, nlt)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force regeneration of all embeddings even if they already exist")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                        help="Number of verses to process in each batch (default: 32)")
    
    args = parser.parse_args()
    
    # Ensure database tables exist
    print("📋 Ensuring database tables exist...")
    await create_tables()
    
    generator = EmbeddingGenerator()
    
    if args.version:
        # Process a specific version
        await generator.generate_embeddings_for_version(
            args.version, 
            batch_size=args.batch_size,
            force_regenerate=args.force
        )
        await generator.verify_embeddings(args.version)
    else:
        # Process all versions
        await generator.generate_all_embeddings(force_regenerate=args.force)
    
    print("✅ Embedding generation complete!")

if __name__ == "__main__":
    asyncio.run(main())
