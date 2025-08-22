# Import FastAPI router and dependencies
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from db.db import get_db
from app.agents.task_router import TaskRouterAgent
from db.models import BibleVerse
from utils.query_validation import is_valid_bible_query

# Initialize the router
router = APIRouter(prefix="/api/bible", tags=["Bible RAG"])

# Initialize task router agent
task_router = TaskRouterAgent()

# Define Pydantic models for request and response validation
# Pydantic models for request/response
class BibleQuery(BaseModel):
    query: str = Field(..., description="Bible reference (e.g., 'John 3:16'), range (e.g., 'Genesis 1:1-3'), or verse quote/topic", min_length=1)
    include_context: bool = Field(default=False, description="Include additional context in response")
    version: str = Field(default="kjv", description="Bible version (kjv, niv, nkjv, nlt)")

class VerseResult(BaseModel):
    reference: str
    book: str
    chapter: int
    verse: int
    text: str

class SemanticResult(BaseModel):
    reference: str
    book: str
    chapter: int
    verse: int
    text: str
    similarity_score: float

class SemanticSearchResponse(BaseModel):
    query: str
    found: bool
    results: List[SemanticResult]
    version: str

class RangeInfo(BaseModel):
    """Information about verse ranges"""
    is_range: bool
    total_verses_requested: Optional[int] = None
    total_verses_found: Optional[int] = None
    missing_verses: Optional[List[str]] = None

class BibleSearchResponse(BaseModel):
    query: str
    query_type: str
    found: bool
    message: str
    results: List[VerseResult]
    next_reference: Optional[str] = None
    ai_response_structured: Optional[Dict[str, Any]] = None
    version: str = "kjv"  # Bible version used for the search
    classification: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    range_info: Optional[RangeInfo] = None  # Range-specific information

class SuggestionsResponse(BaseModel):
    suggestions: List[str]

class UsageExamplesResponse(BaseModel):
    """Examples of different query types"""
    exact_references: List[str]
    verse_ranges: List[str]
    semantic_searches: List[str]
    quoted_text: List[str]

class RangeValidationResponse(BaseModel):
    """Range validation information"""
    valid: bool
    message: str
    range_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BibleQueryWithVersion(BibleQuery):
    version: str = Field(default="kjv", description="Bible version to use (e.g., 'kjv', 'niv', 'nkjv', 'nlt')")

# Define API endpoints for Bible queries and suggestions
@router.post("/search", response_model=BibleSearchResponse)
async def search_bible(
    query: BibleQuery,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for Bible verses using exact reference, ranges, or semantic search
    
    - **query**: Bible reference (e.g., "John 3:16"), range (e.g., "Genesis 1:1-3"), or verse quote/topic
    - **include_context**: Whether to include additional AI context
    - **version**: Bible version to use (e.g., 'kjv', 'niv', 'nkjv', 'nlt')
    
    Examples:
    - Single verse: "John 3:16"
    - Range: "Genesis 1:1-3", "Psalm 23:1-6"
    - Topic: "verses about love"
    - Quote: "For God so loved the world"
    """
    try:
        # Validate query before routing
        if not is_valid_bible_query(query.query):
            return BibleSearchResponse(
                query=query.query,
                query_type="invalid",
                found=False,
                message="This is not a Bible verse.",
                results=[],
                ai_response="",
                classification=None,
                suggestions=None,
                range_info=None
            )
        
        # Validate the version
        from db.models import VERSION_MODELS
        if query.version.lower() not in VERSION_MODELS:
            raise ValueError(f"Unknown Bible version: {query.version}")
        
        # Route the query through the task router with version
        result = await task_router.route_query(db, query.query, query.version.lower())
        
        # Convert database objects to Pydantic models
        verse_results = []
        for verse in result.get("results", []):
            # Accept any object that looks like a verse (works for all version models)
            if verse is None:
                continue
            # Use duck-typing: ensure the object has the expected attributes
            try:
                if isinstance(verse, dict):
                    ref = verse.get('reference')
                    book = verse.get('book')
                    chapter = verse.get('chapter')
                    vnum = verse.get('verse')
                    text = verse.get('text')
                else:
                    ref = getattr(verse, 'reference', None)
                    book = getattr(verse, 'book', None)
                    chapter = getattr(verse, 'chapter', None)
                    vnum = getattr(verse, 'verse', None)
                    text = getattr(verse, 'text', None)

                if ref and book and chapter is not None and vnum is not None and text is not None:
                    verse_results.append(VerseResult(
                        reference=ref,
                        book=book,
                        chapter=int(chapter),
                        verse=int(vnum),
                        text=text
                    ))
            except Exception:
                # Skip any object we can't convert
                continue
        
        # Create range information if applicable
        range_info = None
        if result.get("is_range", False) or (result.get("classification", {}).get("is_range", False)):
            # Check if this was a range query
            range_validation = await task_router.validate_range_input(query.query)
            if range_validation.get("valid", False):
                range_data = range_validation.get("range_info", {})
                total_requested = range_data.get("total_verses", 0)
                total_found = len(verse_results)
                
                # Calculate missing verses
                missing_verses = []
                if total_found < total_requested:
                    found_refs = {verse.reference for verse in verse_results}
                    expected_refs = range_data.get("individual_references", [])
                    missing_verses = [ref for ref in expected_refs if ref not in found_refs]
                
                range_info = RangeInfo(
                    is_range=True,
                    total_verses_requested=total_requested,
                    total_verses_found=total_found,
                    missing_verses=missing_verses if missing_verses else None
                )
            else:
                range_info = RangeInfo(is_range=False)
        
        # Get suggestions for partial matches or no results
        suggestions = []
        if not result["found"] or len(verse_results) == 0:
            suggestions = await task_router.get_suggestions(query.query)

        # Compute next_reference (small, fast DB check) so UI can implement a "Next" button
        next_ref = None
        try:
            if len(verse_results) > 0:
                last = verse_results[-1]
                # Use model-aware lookup to find the immediate next verse in the same version
                from db.models import get_verse_model
                VerseModel = get_verse_model(query.version.lower())

                # Prefer any verse with verse > last.verse in the same chapter (handles duplicate rows)
                qn = select(VerseModel).where(
                    VerseModel.book == last.book,
                    VerseModel.chapter == int(last.chapter),
                    VerseModel.verse > int(last.verse)
                ).order_by(VerseModel.verse).limit(1)
                r = await db.execute(qn)
                nv = r.scalars().first()
                if nv:
                    next_ref = f"{nv.book} {nv.chapter}:{nv.verse}"
                else:
                    # fallback: first verse of the next chapter in same book
                    q2 = select(VerseModel).where(
                        VerseModel.book == last.book,
                        VerseModel.chapter > int(last.chapter)
                    ).order_by(VerseModel.chapter, VerseModel.verse).limit(1)
                    r2 = await db.execute(q2)
                    nv2 = r2.scalars().first()
                    if nv2:
                        next_ref = f"{nv2.book} {nv2.chapter}:{nv2.verse}"
        except Exception:
            # non-fatal; leave next_ref as None on any error
            next_ref = None

        return BibleSearchResponse(
            query=result["query"],
            query_type=result["query_type"],
            found=result["found"],
            message=result["message"],
            results=verse_results,
            next_reference=next_ref,
            ai_response_structured=result.get("ai_response_structured"),
            version=query.version.lower(),
            classification=result.get("classification"),
            suggestions=suggestions if suggestions else None,
            range_info=range_info
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/next", response_model=VerseResult)
async def get_next_verse(
    reference: str,
    version: str = "kjv",
    db: AsyncSession = Depends(get_db)
):
    """
    Get the immediate next verse after the provided reference in the given version.

    Example: /api/bible/next?reference=Matthew%201:1&version=kjv
    """
    try:
        # Validate and parse the reference
        from utils.chunker import BibleChunker
        from db.models import get_verse_model

        chunker = BibleChunker()
        if not chunker.is_reference_format(reference):
            raise HTTPException(status_code=400, detail=f"Invalid reference: {reference}")

        book, chapter, verse = chunker.parse_reference(reference)
        VerseModel = get_verse_model(version.lower())

        # Find next verse within same chapter (verse > current)
        q = select(VerseModel).where(
            VerseModel.book == book,
            VerseModel.chapter == int(chapter),
            VerseModel.verse > int(verse)
        ).order_by(VerseModel.verse).limit(1)
        r = await db.execute(q)
        nv = r.scalars().first()
        if nv:
            return VerseResult(
                reference=nv.reference,
                book=nv.book,
                chapter=nv.chapter,
                verse=nv.verse,
                text=nv.text
            )

        # Fallback: first verse of next chapter in same book
        q2 = select(VerseModel).where(
            VerseModel.book == book,
            VerseModel.chapter > int(chapter)
        ).order_by(VerseModel.chapter, VerseModel.verse).limit(1)
        r2 = await db.execute(q2)
        nv2 = r2.scalars().first()
        if nv2:
            return VerseResult(
                reference=nv2.reference,
                book=nv2.book,
                chapter=nv2.chapter,
                verse=nv2.verse,
                text=nv2.text
            )

        # Explicitly tell clients when there are no more verses
        raise HTTPException(status_code=404, detail=f"No more verses after {reference} in version {version}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get next verse: {str(e)}")


@router.get("/prev", response_model=VerseResult)
async def get_prev_verse(
    reference: str,
    version: str = "kjv",
    db: AsyncSession = Depends(get_db)
):
    """
    Get the immediate previous verse before the provided reference in the given version.
    """
    try:
        from utils.chunker import BibleChunker
        from db.models import get_verse_model

        chunker = BibleChunker()
        if not chunker.is_reference_format(reference):
            raise HTTPException(status_code=400, detail=f"Invalid reference: {reference}")

        book, chapter, verse = chunker.parse_reference(reference)
        VerseModel = get_verse_model(version.lower())

        # Find previous verse within same chapter (verse < current)
        q = select(VerseModel).where(
            VerseModel.book == book,
            VerseModel.chapter == int(chapter),
            VerseModel.verse < int(verse)
        ).order_by(VerseModel.verse.desc()).limit(1)
        r = await db.execute(q)
        pv = r.scalars().first()
        if pv:
            return VerseResult(
                reference=pv.reference,
                book=pv.book,
                chapter=pv.chapter,
                verse=pv.verse,
                text=pv.text
            )

        # Fallback: last verse of previous chapter in same book
        q2 = select(VerseModel).where(
            VerseModel.book == book,
            VerseModel.chapter < int(chapter)
        ).order_by(VerseModel.chapter.desc(), VerseModel.verse.desc()).limit(1)
        r2 = await db.execute(q2)
        pv2 = r2.scalars().first()
        if pv2:
            return VerseResult(
                reference=pv2.reference,
                book=pv2.book,
                chapter=pv2.chapter,
                verse=pv2.verse,
                text=pv2.text
            )

        raise HTTPException(status_code=404, detail=f"No previous verse before {reference} in version {version}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get previous verse: {str(e)}")

@router.get("/search", response_model=BibleSearchResponse)
async def search_bible_get(
    q: str = Query(..., description="Bible reference, range, or verse quote/topic", min_length=1),
    include_context: bool = Query(default=False, description="Include additional context"),
    db: AsyncSession = Depends(get_db)
):
    """
    GET endpoint for Bible search - useful for direct URL access
    
    Examples:
    - /api/bible/search?q=John 3:16
    - /api/bible/search?q=Genesis 1:1-3
    - /api/bible/search?q=verses about love
    """
    query = BibleQuery(query=q, include_context=include_context)
    return await search_bible(query, db)

@router.get("/suggestions", response_model=SuggestionsResponse)
async def get_suggestions(
    q: str = Query(..., description="Partial query to get suggestions for", min_length=1)
):
    """
    Get search suggestions based on partial input
    
    Includes single verses, ranges, and topics
    """
    try:
        suggestions = await task_router.get_suggestions(q)
        return SuggestionsResponse(suggestions=suggestions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

@router.get("/examples", response_model=UsageExamplesResponse)
async def get_usage_examples():
    """
    Get examples of different query types including ranges
    """
    try:
        examples = task_router.get_usage_examples()
        return UsageExamplesResponse(**examples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get examples: {str(e)}")

@router.get("/validate-range", response_model=RangeValidationResponse)
async def validate_range(
    range_query: str = Query(..., description="Range query to validate (e.g., 'Genesis 1:1-3')")
):
    """
    Validate a Bible verse range query
    
    Useful for checking range syntax before making actual search
    """
    try:
        validation_result = await task_router.validate_range_input(range_query)
        return RangeValidationResponse(**validation_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.get("/verse/{reference}", response_model=BibleSearchResponse)
async def get_verse_by_reference(
    reference: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific Bible verse or range by reference
    
    - **reference**: Bible reference like "John 3:16" or range like "Genesis 1:1-3"
    """
    query = BibleQuery(query=reference)
    return await search_bible(query, db)

@router.get("/range/{range_reference}", response_model=BibleSearchResponse)
async def get_verse_range(
    range_reference: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific Bible verse range
    
    - **range_reference**: Bible range like "Genesis 1:1-3", "Psalm 23:1-6"
    """
    # Validate that this is actually a range
    validation = await task_router.validate_range_input(range_reference)
    if not validation.get("valid", False):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid range format: {range_reference}. Use format like 'Genesis 1:1-3'"
        )
    
    query = BibleQuery(query=range_reference)
    return await search_bible(query, db)

@router.get("/books", response_model=List[str])
async def get_bible_books(db: AsyncSession = Depends(get_db)):
    """
    Get list of all Bible books in the database
    """
    try:
        from sqlalchemy import select, distinct
        query = select(distinct(BibleVerse.book)).order_by(BibleVerse.book)
        result = await db.execute(query)
        books = [row[0] for row in result.fetchall()]
        return books
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get books: {str(e)}")

@router.get("/books/{book}/chapters", response_model=List[int])
async def get_book_chapters(
    book: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get list of chapters for a specific book
    """
    try:
        from sqlalchemy import select, distinct
        query = select(distinct(BibleVerse.chapter)).where(
            BibleVerse.book == book
        ).order_by(BibleVerse.chapter)
        result = await db.execute(query)
        chapters = [row[0] for row in result.fetchall()]
        
        if not chapters:
            raise HTTPException(status_code=404, detail=f"Book '{book}' not found")
            
        return chapters
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chapters: {str(e)}")

@router.get("/books/{book}/{chapter}", response_model=List[VerseResult])
async def get_chapter_verses(
    book: str,
    chapter: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get all verses for a specific book and chapter
    """
    try:
        from sqlalchemy import select
        query = select(BibleVerse).where(
            BibleVerse.book == book,
            BibleVerse.chapter == chapter
        ).order_by(BibleVerse.verse)
        
        result = await db.execute(query)
        verses = result.scalars().all()
        
        if not verses:
            raise HTTPException(status_code=404, detail=f"No verses found for {book} {chapter}")
        
        return [
            VerseResult(
                reference=verse.reference,
                book=verse.book,
                chapter=verse.chapter,
                verse=verse.verse,
                text=verse.text
            )
            for verse in verses
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get verses: {str(e)}")

@router.get("/books/{book}/{chapter}/{start_verse}-{end_verse}", response_model=BibleSearchResponse)
async def get_verse_range_by_parts(
    book: str,
    chapter: int,
    start_verse: int,
    end_verse: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a range of verses by individual components
    
    - **book**: Bible book name
    - **chapter**: Chapter number
    - **start_verse**: Starting verse number
    - **end_verse**: Ending verse number
    """
    # Construct the range reference
    range_reference = f"{book} {chapter}:{start_verse}-{end_verse}"
    
    query = BibleQuery(query=range_reference)
    return await search_bible(query, db)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "Bible RAG API",
        "features": {
            "single_verse_lookup": True,
            "verse_range_lookup": True,
            "semantic_search": True,
            "auto_suggestions": True,
            "range_validation": True
        }
    }

@router.post("/semantic", response_model=SemanticSearchResponse)
async def semantic_search(
    query: str = Query(..., description="Text to search for semantically"),
    version: str = Query("kjv", description="Bible version (kjv, niv, nkjv, nlt)"),
    limit: int = Query(3, ge=1, le=10, description="Maximum number of results"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Optional similarity threshold override (0-1)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Perform semantic search and return top results with similarity scores
    """
    try:
        # Route semantic queries through the TaskRouterAgent which supports auto-detection
        print(f"Semantic search request: query='{query}', version='{version}', limit={limit}")

        # Use the TaskRouterAgent to perform semantic search (it handles 'auto')
        try:
            # If a threshold override was provided, set it on the retriever instance
            if threshold is not None:
                try:
                    task_router.retriever_agent._override_min_semantic = float(threshold)
                    print(f"Applied threshold override: {threshold}")
                except Exception:
                    pass

            result = await task_router.route_query(db, query, version.lower())

            # Clear any temporary override
            try:
                task_router.retriever_agent._override_min_semantic = None
                if threshold is not None:
                    print(f"Cleared threshold override after routing: {threshold}")
            except Exception:
                pass

        except Exception as e:
            print(f"Semantic routing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Semantic routing failed: {str(e)}")

        # Expect result to be a dict with 'results' and 'version'
        results = []
        if result.get('results'):
            for r in result.get('results')[:limit]:
                try:
                    # r may be a dict with similarity_score or a DB object
                    if isinstance(r, dict):
                        sim = float(r.get('similarity_score') or 0.0)
                        ref = r.get('reference')
                        book = r.get('book')
                        chapter = int(r.get('chapter')) if r.get('chapter') is not None else 0
                        verse = int(r.get('verse')) if r.get('verse') is not None else 0
                        text = r.get('text')
                    else:
                        sim = float(getattr(r, 'similarity_score', 0.0) or 0.0)
                        ref = getattr(r, 'reference', None)
                        book = getattr(r, 'book', None)
                        chapter = int(getattr(r, 'chapter', 0) or 0)
                        verse = int(getattr(r, 'verse', 0) or 0)
                        text = getattr(r, 'text', None)

                    if ref and text:
                        results.append(SemanticResult(
                            reference=ref,
                            book=book,
                            chapter=chapter,
                            verse=verse,
                            text=text,
                            similarity_score=sim
                        ))
                except Exception:
                    continue

        return SemanticSearchResponse(
            query=result.get('query', query),
            found=bool(results),
            results=results,
            version=result.get('version', version.lower())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Semantic search error: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")












# from fastapi import APIRouter, Depends, HTTPException, Query
# from sqlalchemy.ext.asyncio import AsyncSession
# from pydantic import BaseModel, Field
# from typing import List, Optional, Dict, Any
# from db.db import get_db
# from app.agents.task_router import TaskRouterAgent
# from db.models import BibleVerse
# from utils.query_validation import is_valid_bible_query

# # Initialize the router
# router = APIRouter(prefix="/api/bible", tags=["Bible RAG"])

# # Initialize task router agent
# task_router = TaskRouterAgent()

# # Pydantic models for request/response
# class BibleQuery(BaseModel):
#     query: str = Field(..., description="Bible reference (e.g., 'John 3:16') or verse quote/topic", min_length=1)
#     include_context: bool = Field(default=False, description="Include additional context in response")

# class VerseResult(BaseModel):
#     reference: str
#     book: str
#     chapter: int
#     verse: int
#     text: str

# class BibleSearchResponse(BaseModel):
#     query: str
#     query_type: str
#     found: bool
#     message: str
#     results: List[VerseResult]
#     ai_response: str
#     classification: Optional[Dict[str, Any]] = None
#     suggestions: Optional[List[str]] = None

# class SuggestionsResponse(BaseModel):
#     suggestions: List[str]

# @router.post("/search", response_model=BibleSearchResponse)
# async def search_bible(
#     query: BibleQuery,
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     Search for Bible verses using either exact reference or semantic search
    
#     - **query**: Bible reference (e.g., "John 3:16") or verse quote/topic
#     - **include_context**: Whether to include additional AI context
#     """
#     try:
#         # Validate query before routing
#         if not is_valid_bible_query(query.query):
#             return BibleSearchResponse(
#                 query=query.query,
#                 query_type="invalid",
#                 found=False,
#                 message="This is not a Bible verse.",
#                 results=[],
#                 ai_response="",
#                 classification=None,
#                 suggestions=None
#         )
        
#         # Route the query through the task router
#         result = await task_router.route_query(db, query.query)
        
#         # Convert database objects to Pydantic models
#         verse_results = []
#         for verse in result.get("results", []):
#             if isinstance(verse, BibleVerse):
#                 verse_results.append(VerseResult(
#                     reference=verse.reference,
#                     book=verse.book,
#                     chapter=verse.chapter,
#                     verse=verse.verse,
#                     text=verse.text
#                 ))
        
#         # Get suggestions for partial matches or no results
#         suggestions = []
#         if not result["found"] or len(verse_results) == 0:
#             suggestions = await task_router.get_suggestions(query.query)
        
#         return BibleSearchResponse(
#             query=result["query"],
#             query_type=result["query_type"],
#             found=result["found"],
#             message=result["message"],
#             results=verse_results,
#             ai_response=result.get("ai_response", ""),
#             classification=result.get("classification"),
#             suggestions=suggestions if suggestions else None
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# @router.get("/search", response_model=BibleSearchResponse)
# async def search_bible_get(
#     q: str = Query(..., description="Bible reference or verse quote/topic", min_length=1),
#     include_context: bool = Query(default=False, description="Include additional context"),
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     GET endpoint for Bible search - useful for direct URL access
#     """
#     query = BibleQuery(query=q, include_context=include_context)
#     return await search_bible(query, db)

# @router.get("/suggestions", response_model=SuggestionsResponse)
# async def get_suggestions(
#     q: str = Query(..., description="Partial query to get suggestions for", min_length=1)
# ):
#     """
#     Get search suggestions based on partial input
#     """
#     try:
#         suggestions = await task_router.get_suggestions(q)
#         return SuggestionsResponse(suggestions=suggestions)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

# @router.get("/verse/{reference}", response_model=BibleSearchResponse)
# async def get_verse_by_reference(
#     reference: str,
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     Get a specific Bible verse by reference
    
#     - **reference**: Bible reference like "John 3:16"
#     """
#     query = BibleQuery(query=reference)
#     return await search_bible(query, db)

# @router.get("/books", response_model=List[str])
# async def get_bible_books(db: AsyncSession = Depends(get_db)):
#     """
#     Get list of all Bible books in the database
#     """
#     try:
#         from sqlalchemy import select, distinct
#         query = select(distinct(BibleVerse.book)).order_by(BibleVerse.book)
#         result = await db.execute(query)
#         books = [row[0] for row in result.fetchall()]
#         return books
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get books: {str(e)}")

# @router.get("/books/{book}/chapters", response_model=List[int])
# async def get_book_chapters(
#     book: str,
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     Get list of chapters for a specific book
#     """
#     try:
#         from sqlalchemy import select, distinct
#         query = select(distinct(BibleVerse.chapter)).where(
#             BibleVerse.book == book
#         ).order_by(BibleVerse.chapter)
#         result = await db.execute(query)
#         chapters = [row[0] for row in result.fetchall()]
        
#         if not chapters:
#             raise HTTPException(status_code=404, detail=f"Book '{book}' not found")
            
#         return chapters
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get chapters: {str(e)}")

# @router.get("/books/{book}/{chapter}", response_model=List[VerseResult])
# async def get_chapter_verses(
#     book: str,
#     chapter: int,
#     db: AsyncSession = Depends(get_db)
# ):
#     """
#     Get all verses for a specific book and chapter
#     """
#     try:
#         from sqlalchemy import select
#         query = select(BibleVerse).where(
#             BibleVerse.book == book,
#             BibleVerse.chapter == chapter
#         ).order_by(BibleVerse.verse)
        
#         result = await db.execute(query)
#         verses = result.scalars().all()
        
#         if not verses:
#             raise HTTPException(status_code=404, detail=f"No verses found for {book} {chapter}")
        
#         return [
#             VerseResult(
#                 reference=verse.reference,
#                 book=verse.book,
#                 chapter=verse.chapter,
#                 verse=verse.verse,
#                 text=verse.text
#             )
#             for verse in verses
#         ]
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to get verses: {str(e)}")

# @router.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "service": "Bible RAG API"}