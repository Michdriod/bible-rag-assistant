#!/usr/bin/env python3
"""
Direct database test to check if Bible data exists
"""
import asyncio
import asyncpg
from db.db import get_database_url

async def test_database():
    """Test if database contains Bible data"""
    try:
        # Get database URL
        db_url = get_database_url()
        
        # Connect directly using asyncpg
        conn = await asyncpg.connect(db_url)
        
        # Check if tables exist
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('kjv', 'niv', 'nkjv', 'nlt')
        """)
        
        print("Available Bible tables:")
        for table in tables:
            print(f"  - {table['table_name']}")
        
        # Check if KJV table has data
        if any(t['table_name'] == 'kjv' for t in tables):
            count = await conn.fetchval("SELECT COUNT(*) FROM kjv")
            print(f"\nKJV table has {count} verses")
            
            # Get a sample verse
            sample = await conn.fetchrow("SELECT reference, text FROM kjv LIMIT 1")
            if sample:
                print(f"Sample verse: {sample['reference']} - {sample['text'][:100]}...")
            
            # Try to find the specific text
            search_results = await conn.fetch("""
                SELECT reference, text, 
                       CASE WHEN LOWER(text) LIKE LOWER($1) THEN 'exact'
                            WHEN LOWER(text) LIKE '%blessed%' AND LOWER(text) LIKE '%revile%' THEN 'partial'
                            ELSE 'none' END as match_type
                FROM kjv 
                WHERE LOWER(text) LIKE '%blessed%' 
                   OR LOWER(text) LIKE '%revile%'
                   OR LOWER(text) LIKE '%persecute%'
                ORDER BY match_type DESC
                LIMIT 5
            """, "%blessed are ye, when men shall revile you%")
            
            print("\nSearching for 'blessed are ye, when men shall revile you':")
            for result in search_results:
                print(f"  {result['reference']}: {result['text'][:150]}... ({result['match_type']} match)")
        
        await conn.close()
        
    except Exception as e:
        print(f"Database test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_database())
