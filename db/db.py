import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from db.models import Base
from dotenv import load_dotenv
from sqlalchemy import text

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:michwaleh@localhost:5432/Bible_RAG")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Set to False in production
    future=True,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=300,    # Recycle connections every 5 minutes
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_db():
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def create_tables():
    """Create all tables and indexes"""
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        
        # Get the list of Bible versions
        from db.models import VERSION_MODELS
        
        # Create indexes for each version
        for version, model_class in VERSION_MODELS.items():
            table_name = model_class.__tablename__
            indexes = [
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_reference ON {table_name}(reference)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_book_chapter_verse ON {table_name}(book, chapter, verse)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_book ON {table_name}(book)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_chapter ON {table_name}(chapter)",
            ]
            
            for index_sql in indexes:
                try:
                    await conn.execute(text(index_sql))
                    print(f"✅ Created index for {version}: {index_sql.split('idx_')[1].split(' ')[0] if 'idx_' in index_sql else 'unknown'}")
                except Exception as e:
                    print(f"⚠️  Index creation warning for {version}: {e}")

async def create_vector_index():
    """Create vector similarity index - should be done after data ingestion"""
    async with engine.begin() as conn:
        # Get the list of Bible versions
        from db.models import VERSION_MODELS
        
        for version, model_class in VERSION_MODELS.items():
            table_name = model_class.__tablename__
            try:
                # This should be created after you have data in the table
                # The lists parameter should be approximately sqrt(total_rows)
                await conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding "
                    f"ON {table_name} USING ivfflat (embedding vector_cosine_ops) "
                    f"WITH (lists = 100)"
                ))
                print(f"✅ Created vector similarity index for {version}")
            except Exception as e:
                print(f"⚠️  Vector index creation warning for {version}: {e}")
                print(f"Note: Vector index for {version} should be created after data ingestion")

async def drop_tables():
    """Drop all tables - useful for development"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        print("🗑️  Dropped all tables")

async def get_table_stats():
    """Get statistics about all Bible version tables"""
    async with AsyncSessionLocal() as session:
        try:
            # Get the list of Bible versions
            from db.models import VERSION_MODELS
            
            # Initialize results dictionary
            stats = {}
            
            # Get stats for each version
            for version, model_class in VERSION_MODELS.items():
                table_name = model_class.__tablename__
                
                # Check if table exists and get row count
                result = await session.execute(text(
                    f"SELECT COUNT(*) FROM {table_name}"
                ))
                verse_count = result.scalar() or 0
                
                # Get unique books count
                result = await session.execute(text(
                    f"SELECT COUNT(DISTINCT book) FROM {table_name}"
                ))
                book_count = result.scalar() or 0
                
                # Check if vector index exists
                result = await session.execute(text(
                    f"SELECT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_{table_name}_embedding')"
                ))
                has_vector_index = result.scalar()
                
                stats[version] = {
                    "total_verses": verse_count,
                    "unique_books": book_count,
                    "has_vector_index": has_vector_index
                }
            
            # Add overall stats
            total_verses = sum(v["total_verses"] for v in stats.values())
            stats["overall"] = {
                "total_verses": total_verses,
                "versions_available": [v for v, s in stats.items() if s["total_verses"] > 0]
            }
            
            return stats
        except Exception as e:
            print(f"Error getting table stats: {e}")
            return {}

# Import necessary modules and libraries for database operations
# Define database configuration and create async engine
# Create async session factory and dependency for database session
# Define function to create tables and indexes for Bible versions
