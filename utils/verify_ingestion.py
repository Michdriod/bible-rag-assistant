# Import necessary modules and libraries for ingestion verification
import asyncio
import numpy as np
from sqlalchemy import text, func, select
from db.db import AsyncSessionLocal
from db.models import BibleVerse, get_verse_model

# Define supported Bible versions
SUPPORTED_VERSIONS = ["kjv", "niv", "nkjv", "nlt"]

# Implement a function to verify Bible ingestion for a specific version
async def verify_bible_ingestion(version="kjv"):
    """
    Comprehensive verification of Bible ingestion for a specific version
    
    Args:
        version: Bible version to verify (e.g., "kjv", "niv", "nkjv", "nlt")
    """
    print(f"🔍 Verifying Bible ingestion for {version.upper()}...")
    
    # Get the appropriate table name for the version
    table_name = f"bible_verses_{version}"
    if version == "kjv":  # Legacy support for the original table
        table_name = "bible_verses"
    
    try:
        async with AsyncSessionLocal() as session:
            # 1. Basic count check
            print(f"\n📊 Basic Statistics for {version.upper()}:")
            
            # Total verses
            total_result = await session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            total_count = total_result.scalar()
            print(f"   Total verses: {total_count}")
            
            # Unique books
            books_result = await session.execute(text(f"SELECT COUNT(DISTINCT book) FROM {table_name}"))
            books_count = books_result.scalar()
            print(f"   Unique books: {books_count}")
            
            # Unique chapters
            chapters_result = await session.execute(text(f"SELECT COUNT(DISTINCT CONCAT(book, '-', chapter)) FROM {table_name}"))
            chapters_count = chapters_result.scalar()
            print(f"   Unique chapters: {chapters_count}")
            
            # 2. Sample data check
            print(f"\n📖 Sample Verses from {version.upper()}:")
            sample_result = await session.execute(text(f"""
                SELECT reference, LEFT(text, 80) as text_preview, 
                       array_length(embedding, 1) as embedding_dim
                FROM {table_name} 
                ORDER BY book, chapter, verse 
                LIMIT 3
            """))
            
            for row in sample_result.fetchall():
                print(f"   {row.reference}: {row.text_preview}...")
                print(f"      Embedding dimension: {row.embedding_dim}")
            
            # 3. Book distribution
            print(f"\n📚 Book Distribution for {version.upper()} (first 10):")
            books_dist_result = await session.execute(text(f"""
                SELECT book, COUNT(*) as verse_count 
                FROM {table_name} 
                GROUP BY book 
                ORDER BY MIN(chapter), MIN(verse)
                LIMIT 10
            """))
            
            for row in books_dist_result.fetchall():
                print(f"   {row.book}: {row.verse_count} verses")
            
            # 4. Embedding validation
            print(f"\n🧠 Embedding Validation for {version.upper()}:")
            embedding_check = await session.execute(text(f"""
                SELECT 
                    COUNT(*) as total_embeddings,
                    COUNT(CASE WHEN embedding IS NULL THEN 1 END) as null_embeddings,
                    AVG(array_length(embedding, 1)) as avg_dimension
                FROM {table_name}
            """))
            
            emb_row = embedding_check.fetchone()
            print(f"   Total embeddings: {emb_row.total_embeddings}")
            print(f"   Null embeddings: {emb_row.null_embeddings}")
            print(f"   Average dimension: {emb_row.avg_dimension:.1f}")
            
            # 5. Test specific references
            print(f"\n🎯 Testing Specific References in {version.upper()}:")
            test_refs = ["Genesis 1:1", "John 3:16", "Psalm 23:1", "Matthew 5:3"]
            
            for ref in test_refs:
                ref_result = await session.execute(
                    text(f"SELECT reference, LEFT(text, 60) as text FROM {table_name} WHERE reference = :ref"),
                    {"ref": ref}
                )
                ref_row = ref_result.fetchone()
                if ref_row:
                    print(f"   ✅ {ref_row.reference}: {ref_row.text}...")
                else:
                    print(f"   ❌ {ref}: Not found")
            
            # 6. Database indexes check
            print(f"\n🗂️  Index Status for {version.upper()}:")
            index_result = await session.execute(text("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE tablename = :table_name
                ORDER BY indexname
            """), {"table_name": table_name})
            
            for row in index_result.fetchall():
                print(f"   ✅ {row.indexname}")
            
            # 7. Quick similarity test (if we have embeddings)
            if emb_row.null_embeddings == 0 and total_count > 0:
                print(f"\n🔍 Testing Semantic Search for {version.upper()}:")
                
                # Get a sample embedding for testing
                sample_emb_result = await session.execute(text(f"""
                    SELECT embedding FROM {table_name} WHERE reference = 'John 3:16' LIMIT 1
                """))
                sample_emb_row = sample_emb_result.fetchone()
                
                if sample_emb_row:
                    # Test similarity search with the same embedding (should return high similarity)
                    similarity_result = await session.execute(text(f"""
                        SELECT reference, 1 - (embedding <=> :test_embedding) as similarity
                        FROM {table_name} 
                        ORDER BY embedding <=> :test_embedding
                        LIMIT 3
                    """), {"test_embedding": sample_emb_row.embedding})
                    
                    print(f"   Top 3 most similar to 'John 3:16' in {version.upper()}:")
                    for row in similarity_result.fetchall():
                        print(f"      {row.reference}: {row.similarity:.3f}")
            
            # 8. Summary
            print(f"\n✅ Verification Summary for {version.upper()}:")
            print(f"   Database contains {total_count} verses from {books_count} books")
            print(f"   All embeddings present: {'Yes' if emb_row.null_embeddings == 0 else 'No'}")
            print(f"   Embedding dimension consistent: {'Yes' if emb_row.avg_dimension == 384 else f'No (avg: {emb_row.avg_dimension})'}")
            
            if total_count > 30000:
                print(f"   ✅ Full Bible appears to be loaded for {version.upper()}")
            elif total_count > 1000:
                print(f"   ⚠️  Partial Bible loaded for {version.upper()}")
            else:
                print(f"   ❌ Very few verses loaded for {version.upper()}")
                
            return True
            
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def quick_test_search(version="kjv"):
    """
    Quick test of search functionality for a specific version
    
    Args:
        version: Bible version to test (e.g., "kjv", "niv", "nkjv", "nlt")
    """
    print(f"\n🔍 Quick Search Test for {version.upper()}:")
    
    # Get the appropriate table name for the version
    table_name = f"bible_verses_{version}"
    if version == "kjv":  # Legacy support for the original table
        table_name = "bible_verses"
    
    try:
        async with AsyncSessionLocal() as session:
            # Test exact reference search
            exact_result = await session.execute(
                text(f"SELECT reference, text FROM {table_name} WHERE reference = 'John 3:16'")
            )
            exact_row = exact_result.fetchone()
            
            if exact_row:
                print(f"✅ Exact search works in {version.upper()}: {exact_row.reference}")
                print(f"   Text: {exact_row.text[:100]}...")
            else:
                print(f"❌ Exact search failed in {version.upper()} - John 3:16 not found")
            
            # Test text search
            text_search_result = await session.execute(text(f"""
                SELECT reference, LEFT(text, 60) as text_preview
                FROM {table_name} 
                WHERE text ILIKE '%love%' 
                LIMIT 3
            """))
            
            print(f"✅ Text search results for 'love' in {version.upper()}:")
            for row in text_search_result.fetchall():
                print(f"   {row.reference}: {row.text_preview}...")
                
    except Exception as e:
        print(f"❌ Search test failed for {version.upper()}: {str(e)}")

async def verify_all_versions():
    """Verify ingestion for all supported Bible versions"""
    print("🚀 Starting Bible Ingestion Verification for All Versions\n")
    
    results = {}
    for version in SUPPORTED_VERSIONS:
        print(f"\n{'='*50}")
        print(f"🔍 VERIFYING {version.upper()} BIBLE")
        print(f"{'='*50}")
        
        success = await verify_bible_ingestion(version)
        await quick_test_search(version)
        
        results[version] = success
    
    print("\n" + "="*80)
    print("📊 VERIFICATION SUMMARY")
    print("="*80)
    
    for version, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{version.upper()}: {status}")
    
    return all(results.values())

async def main():
    """Run all verification checks"""
    print("🚀 Starting Bible Ingestion Verification\n")
    
    # Check if all versions are requested
    success = await verify_all_versions()
    
    if success:
        await quick_test_search()
        print("\n🎉 Verification completed successfully!")
    else:
        print("\n💥 Verification failed!")
    
    return success

async def compare_versions(version1="kjv", version2="niv", sample_size=5):
    """
    Compare verses between two Bible versions
    
    Args:
        version1: First Bible version (e.g., "kjv")
        version2: Second Bible version (e.g., "niv")
        sample_size: Number of verses to compare
    """
    print(f"\n🔄 Comparing {version1.upper()} and {version2.upper()} (sample of {sample_size} verses)")
    
    # Get the appropriate table names
    table1 = "bible_verses" if version1 == "kjv" else f"bible_verses_{version1}"
    table2 = "bible_verses" if version2 == "kjv" else f"bible_verses_{version2}"
    
    try:
        async with AsyncSessionLocal() as session:
            # Get some common references that exist in both versions
            ref_result = await session.execute(text(f"""
                SELECT a.reference 
                FROM {table1} a
                JOIN {table2} b ON a.reference = b.reference
                ORDER BY RANDOM()
                LIMIT :limit
            """), {"limit": sample_size})
            
            references = [row.reference for row in ref_result.fetchall()]
            
            if not references:
                print(f"❌ No common references found between {version1.upper()} and {version2.upper()}")
                return
            
            # Compare the text of these references
            print(f"\n📊 Verse Comparison ({version1.upper()} vs {version2.upper()}):")
            
            for ref in references:
                v1_result = await session.execute(
                    text(f"SELECT text FROM {table1} WHERE reference = :ref"),
                    {"ref": ref}
                )
                v2_result = await session.execute(
                    text(f"SELECT text FROM {table2} WHERE reference = :ref"),
                    {"ref": ref}
                )
                
                v1_text = v1_result.fetchone()
                v2_text = v2_result.fetchone()
                
                if v1_text and v2_text:
                    print(f"\n📖 {ref}:")
                    print(f"  {version1.upper()}: {v1_text.text[:100]}..." if len(v1_text.text) > 100 else v1_text.text)
                    print(f"  {version2.upper()}: {v2_text.text[:100]}..." if len(v2_text.text) > 100 else v2_text.text)
    
    except Exception as e:
        print(f"❌ Version comparison failed: {str(e)}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    if not args:
        print("Running verification for all versions")
        asyncio.run(main())
    elif len(args) == 1 and args[0] == "compare":
        print("Running version comparison")
        asyncio.run(compare_versions("kjv", "niv"))
    elif len(args) == 1:
        # Verify a specific version
        version = args[0].lower()
        if version in SUPPORTED_VERSIONS:
            print(f"Verifying {version.upper()} version only")
            asyncio.run(verify_bible_ingestion(version))
            asyncio.run(quick_test_search(version))
        else:
            print(f"Unsupported version: {version}")
            print(f"Supported versions: {', '.join(SUPPORTED_VERSIONS)}")
    elif len(args) == 3 and args[0] == "compare":
        # Compare two specific versions
        v1, v2 = args[1].lower(), args[2].lower()
        if v1 in SUPPORTED_VERSIONS and v2 in SUPPORTED_VERSIONS:
            asyncio.run(compare_versions(v1, v2))
        else:
            print(f"One or both versions not supported: {v1}, {v2}")
            print(f"Supported versions: {', '.join(SUPPORTED_VERSIONS)}")
    else:
        print("Usage:")
        print("  python verify_ingestion.py              # Verify all versions")
        print("  python verify_ingestion.py kjv          # Verify specific version")
        print("  python verify_ingestion.py compare      # Compare KJV and NIV")
        print("  python verify_ingestion.py compare kjv niv  # Compare specific versions")