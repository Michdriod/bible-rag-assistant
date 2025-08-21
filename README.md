## Bible RAG Agent System

A focused Retrieval-Augmented Generation (RAG) application for searching Bible verses. It provides exact-reference lookups, range handling, semantic search (vector embeddings), a web UI with presentation mode, and a small set of backend APIs.

## At a glance
- FastAPI backend (served from `app.main`) with an API router at `/api/bible`.
- PostgreSQL database (recommended with `pgvector`) for verse storage and embeddings.
- Vanilla JavaScript frontend in `static/` (`index.html`, `app.js`, `presentation.html`).
- Features: exact references, verse ranges, smart chunked pagination for long passages, semantic search with top-N candidates, voice-capture (Web Speech API), and fullscreen presentation mode.

## Quick start (development)

Prerequisites
- Python 3.12+ (the project is tested on 3.12)
- PostgreSQL database with the `pgvector` extension
- A POSIX shell (macOS `zsh` used in examples)

Local setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies (project tracks deps in `pyproject.toml`):

```bash
pip install -r requiremnets.txt
```

3. Create a `.env` file in the project root with at minimum a `DATABASE_URL` (do not commit this file):

```text
DATABASE_URL=postgresql+asyncpg://<user>:<password>@localhost:5432/<db_name>
```

4. Start the development server:

```bash
# run from repo root
export DEBUG=true   # optional
uvicorn app.main:app --reload
```

Open the UI: http://127.0.0.1:8000

## Important env notes
- Keep secrets (API keys, DB credentials) out of the repo. Add `.env` to `.gitignore`.
- If you accidentally commit secrets, rotate them immediately and remove them from history (see Troubleshooting / Git hygiene below).

## Backend API overview
The FastAPI app mounts routes under `/api/bible`. Major endpoints used by the frontend include:

- POST /api/bible/search
   - Request JSON: { "query": "John 3:16" , "version": "kjv", "include_context": false }
   - Supports exact references, ranges (e.g., `Genesis 1:1-12`), free-text/topic, and quoted text.
   - Returns structured `results` (list of verse objects), a `message`, optional `ai_response_structured`, and a computed `next_reference` for UI navigation.

- GET /api/bible/search?q=... (convenience GET)

- GET /api/bible/next?reference=Book+X:Y:Z&version=kjv
   - Returns the immediate next verse after the provided reference.

- GET /api/bible/prev?reference=...&version=...
   - Returns the immediate previous verse.

- GET /api/bible/suggestions?q=... and /api/bible/examples (helpers for UI)

- Health: GET /health

Note: The router also exposes range validation and other helper endpoints used by the UI.

Semantic search
- The project includes a semantic search flow that uses vector embeddings stored in the DB (pgvector). The frontend exposes a `semantic` search mode which calls a semantic endpoint and shows the top-3 candidate verses; the user can click one to select it and then continue using Next/Prev navigation.

Example usage (curl):

```bash
curl -s -X POST "http://127.0.0.1:8000/api/bible/search" \
   -H "Content-Type: application/json" \
   -d '{"query":"Genesis 1:1-12","version":"kjv"}' | jq
```

## Frontend behavior (what to expect)

- Main UI: `static/index.html` + `static/app.js`.
- Presentation mode: `static/presentation.html` (loads the same logic but optimized for projector display).
- Smart chunking: when a range returns 4+ verses the frontend uses "smart chunking" (default chunk size = 3). The UI will show the first chunk and Next/Prev move through chunks; when you reach the end of the range, Next continues to the verse after the range using the backend `/next` endpoint. Similarly Prev moves before the range when at the first chunk.
- Canonical reference in the top `#reference` element is computed from API results (the app prefers a canonical display like `Genesis 1:1-3`).
- Voice capture: a microphone button uses the browser Web Speech API (when available). Transcribed speech is heuristically classified as a reference (e.g., `John 3:16`) or natural language; the app then triggers a reference lookup or a semantic search automatically.

UX details
- Short ranges (2-3 verses) are shown without chunking and use a paged view by verses-per-page setting.
- Single verses are displayed plainly.
- Semantic search shows up to three candidates with similarity scores and requires the user to click a result to set it as the active reference.

## Embeddings and ingestion

- `embed.py` and `utils/generate_embeddings.py` are provided to create/store embeddings for verse rows already present in the DB.
- Embedding scripts will usually only process rows where `embedding IS NULL` (no OFFSET) so re-running won't reprocess already-embedded rows unless `--force` is used.

## Tests

- Unit/integration tests are located in `tests/`. Run with pytest:

```bash
pip install pytest httpx
pytest -q
```

## Troubleshooting & developer tips

- Static assets caching: if you change `static/app.js` and see stale behavior, do a hard reload (Shift+Cmd+R on mac) or open DevTools → Network → Disable cache.
- If presentation/fullscreen behaves differently, ensure the browser loaded the latest `app.js` and that no duplicate event handlers are present.
- If semantic results look empty, confirm the DB has embeddings and that `pgvector` is configured correctly.

Git hygiene (secrets)
- Never commit `.env` or API keys. If you accidentally push secrets to GitHub, rotate the exposed key immediately and remove it from history using `git filter-repo` or the BFG tool. Add `.env` to `.gitignore`.

Example quick fix for a recent accidental commit (safe when the bad commit is the latest):

```bash
# move sensitive file out of repo
mv .env ../env-backup
git rm --cached .env || true
echo ".env" >> .gitignore
git add .gitignore
git commit --amend --no-edit
git push --force-with-lease origin master
```

If the secret exists in older commits, prefer `git filter-repo` (recommended) or the BFG Cleaner; coordinate with collaborators because history rewrite requires force-push and re-clone.

## Contributing and next steps

- If you'd like, I can:
   - Add CI checks to block accidental secrets locally
   - Add automated tests for chunking/navigation and semantic selection
   - Add an optional fingerprinted static build step for production

## Contact / Support

Open an issue in this repository with reproduction steps and example inputs (e.g., which reference or natural-language query produced unexpected results). Include browser console logs for frontend issues.

---
Generated from the current codebase (FastAPI router: `app/rag_router.py`, server entry: `app/main.py`, frontend: `static/app.js`).

