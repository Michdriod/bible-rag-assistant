# Import necessary modules and libraries for the application
import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv
import logging

from app.rag_router import router as rag_router
from db.db import create_tables, engine

# Load environment variables
load_dotenv()

# Configure root logging for debug during local runs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s [%(name)s] %(message)s')
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    print("🚀 Starting Bible RAG Assistant...")
    await create_tables()
    print("✅ Database tables created/verified")
    print("🎯 Bible RAG API is ready!")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Bible RAG Assistant...")
    # Attempt to clean up common multiprocessing/executor resources that may leak semaphores
    try:
        # If an executor was stored on app.state, shut it down
        ex = getattr(app.state, 'process_executor', None)
        if ex is not None:
            try:
                ex.shutdown(wait=True)
                print('✅ process_executor shutdown')
            except Exception:
                print('⚠️ process_executor shutdown failed')

        mgr = getattr(app.state, 'mp_manager', None)
        if mgr is not None:
            try:
                mgr.shutdown()
                print('✅ mp_manager shutdown')
            except Exception:
                print('⚠️ mp_manager shutdown failed')

        # Try to join/terminate any active multiprocessing children
        import multiprocessing
        children = multiprocessing.active_children()
        if children:
            print(f'⚠️ Found {len(children)} active child process(es); attempting to terminate...')
            for c in children:
                try:
                    c.terminate()
                except Exception:
                    pass
    except Exception as e:
        print('⚠️ Exception during shutdown cleanup:', e)

    await engine.dispose()

# Create FastAPI app
app = FastAPI(
    title="Bible RAG Assistant",
    description="A Retrieval-Augmented Generation (RAG) system for Bible verse search and study",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(rag_router)

# Setup templates for frontend
templates = Jinja2Templates(directory="static")

# Serve static files (CSS, JS)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main frontend page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/docs-redirect")
async def docs_redirect():
    """Redirect to API documentation"""
    return {"message": "Visit /docs for API documentation"}

@app.get("/health")
async def root_health():
    """Root health check"""
    return {
        "status": "healthy",
        "message": "Bible RAG Assistant API",
        "version": "1.0.0",
        "docs": "/docs"
    }

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )