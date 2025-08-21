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
   # Bible RAG Agent System

   A small Retrieval-Augmented Generation (RAG) system for searching Bible verses. Designed to work with a PostgreSQL database (with pgvector) and a lightweight FastAPI frontend for live presentation use.

   ## Highlights

   - Exact and range lookups (e.g. `John 3:16`, `Matthew 5:3-12`)
   - Semantic search via vector embeddings
   - Multiple versions supported: `kjv`, `niv`, `nkjv`, `nlt`
   - Deterministic structured AI output (no LLM network calls by default)
   - Presentation mode for full-screen projector display with keyboard navigation

   ## Quick start

   Requirements
   - Python 3.10+ (3.12 tested in this workspace)
   - PostgreSQL with `pgvector` extension
   - A Python virtualenv (recommended)

   Install

   ```bash
   # from project root
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requiremnets.txt
   ```

   Environment

   Create a `.env` file in the project root with at least your database URL:

   ```
   DATABASE_URL=postgresql+asyncpg://<user>:<pass>@localhost:5432/<db>
   ```

   Run the app (development)

   ```bash
   export DEBUG=true      # optional, enables extra logs
   uvicorn app.main:app --reload
   ```

   Open the UI: http://127.0.0.1:8000

   ## Embeddings

   If your Bible text is already in the database, generate embeddings without re-ingesting:

   ```bash
   # interactive script
   python embed.py

   # or use the generator script for specific versions
   python -m utils.generate_embeddings --version kjv
   python -m utils.generate_embeddings   # all versions
   python -m utils.generate_embeddings --force  # regenerate
   ```

   Notes
   - The generator uses a `WHERE embedding IS NULL LIMIT <batch>` loop (no OFFSET) so it won't re-process already-embedded rows.
   - The code stores vectors in a pgvector-compatible form. Ensure `pgvector` is installed in your DB.

   ## Presentation mode

   Open the presentation page in a separate window (projector) to display a single verse in large font and navigate with keyboard arrows or the on-screen buttons:

   ```
   http://127.0.0.1:8000/static/presentation.html?reference=John%203:16&version=kjv
   ```

   You can also open a verse from the main UI using the Presentation button on the verse detail view.

   ## Cache-busting (development)

   Browsers aggressively cache static assets. During development the app now injects a timestamp query parameter to ensure fresh `app.js` is fetched on each load. If you still see stale UI:

   - Hard-reload (Shift+Cmd+R on mac)
   - Open DevTools → Network → check "Disable cache" and reload
   - Or use an Incognito window

   ## Tests

   Basic integration tests are included in `tests/`. They require `pytest` and `httpx`:

   ```bash
   pip install pytest httpx
   pytest -q
   ```

   ## Troubleshooting

   resource_tracker warning on shutdown

   If you see a warning like:

   ```
   resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
   ```

   It usually means a multiprocessing/executor/manager wasn't shut down before process exit. Fixes:

   - Run without `--reload` to see if the dev reloader caused it.
   - Make sure any `ProcessPoolExecutor`, `multiprocessing.Manager`, or `Pool` you create is `.shutdown()`/`.close()`/`.join()` in an `@app.on_event('shutdown')` handler.

   This project includes defensive shutdown cleanup in `app/main.py` to attempt to close common executors/managers and terminate stray child processes on shutdown.

   ## Deployment notes

   - Use a proper process manager (systemd, docker, or a cloud service) in production and avoid uvicorn `--reload` there.
   - Consider fingerprinting static assets in production instead of timestamp query strings.

   ## Want changes?

   Tell me whether you prefer a single-file frontend (merge presentation into `app.js`) or keep separate pages (current). I can switch either way and adapt cache-busting/fingerprinting accordingly.
