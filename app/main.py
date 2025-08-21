# Import necessary modules and libraries for the application
import os
from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from dotenv import load_dotenv

from app.rag_router import router as rag_router
from db.db import create_tables, engine

# Load environment variables
load_dotenv()

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