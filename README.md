# Bible RAG Agent System

A Retrieval-Augmented Generation (RAG) system for Bible verses with support for multiple Bible versions.

## Features

- Search Bible verses by reference, range, topic, or semantic meaning
- Support for multiple Bible versions (KJV, NIV, NKJV, NLT)
- Semantic search using vector embeddings
- Web interface for querying Bible verses
- CLI interface for command-line access

## Setup

### Prerequisites

- Python 3.9+
- PostgreSQL with pgvector extension
- Python packages (see requirements.txt)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file:

   ```bash
   DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/bible_rag
   GROQ_API_KEY=your_groq_api_key
   ```
4. Generate embeddings for Bible verses (see below)

## Generating Embeddings

If your Bible data is already in the database, you can generate embeddings using:

```bash
# Simple CLI interface
python embed.py

# OR command line options
python -m utils.generate_embeddings --version kjv  # Generate for KJV only
python -m utils.generate_embeddings               # Generate for all versions
python -m utils.generate_embeddings --force       # Regenerate all embeddings
```

## Usage

### Web Interface

1. Start the web server:

   ```bash
   uvicorn app.main:app --reload
   ```
2. Open [http://localhost:8000](http://localhost:8000) in your browser
3. Use the search box to query Bible verses

### CLI Interface

```bash
python cli.py "John 3:16"        # Lookup by reference
python cli.py "verses about love" # Search by topic
```

## Multiple Bible Versions

The system supports the following Bible versions:

- KJV (King James Version)
- NIV (New International Version)
- NKJV (New King James Version)
- NLT (New Living Translation)

Select your preferred version from the dropdown menu in the web interface or specify it in the CLI.
